# Sirius — Realtime Fraud Scoring API

A RESTful service that wraps a pre-trained `sklearn.ensemble.RandomForestClassifier` and returns the
probability that a payment transaction is fraudulent.

The API returns a **probability rather than a fraud/legitimate decision**. The threshold for making that decision is a business choice and can change depending on risk appetite, merchant type, seasonality, etc. Keeping the threshold outside the model service means that changing the policy does not require redeploying the model.

---

## Contents

- [Quick start](#quick-start)
- [API](#api)
- [The model](#the-model)
- [Performance](#performance)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Feature computation and serving](#feature-computation-and-serving)
- [Testing](#testing)
- [Other considerations](#other-considerations)
- [Given more time](#given-more-time)

---

## Quick start

### Docker (recommended)

```bash
docker build -t sirius:local .
docker run --rm -p 8888:8888 sirius:local
```

### Local

Requires Python 3.10 and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8888
```

Either way the service listens on `http://localhost:8888`, with interactive OpenAPI docs at
`http://localhost:8888/docs`.

Verify with the supplied mock request body:

```bash
curl -X POST http://localhost:8888/fraud-score \
  -H "Content-Type: application/json" \
  -d @tests/mock-request-body.json
```

### Tests

```bash
poetry install --with dev
poetry run pytest
```

### Dependency pinning

The assignment pins `scikit-learn==1.0.2`, which constrains everything downstream of it. `numpy<1.24`
and `pandas<2.3` are required because sklearn 1.0.2 predates the `np.float`/`np.int` alias removal in
NumPy 1.24, and Python is capped at `<3.11` because no 1.0.2 wheels were built for 3.11+. These are
not arbitrary — relaxing any one of them breaks unpickling of the model binary.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/fraud-model.pickle` | Location of the serialised model |

Read from the environment, with `.env` loaded at startup for local development.

---

## API

### `POST /fraud-score`

Request body — all 30 features are required, all floats:

```json
{
  "Time": 0.0,
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  "...": "V3 through V28",
  "Amount": 149.62
}
```

Response:

```json
{
  "fraud-score": 0.086
}
```

`fraud-score` is `P(class = 1)`, a float in `[0.0, 1.0]`. The hyphenated key follows the brief; it is
carried by a Pydantic alias, since `fraud-score` is not a valid Python identifier.

| Status | Meaning |
|---|---|
| `200` | Score returned |
| `422` | Request body failed validation — missing, extra or non-numeric fields |
| `500` | Model inference failed; details logged server-side, not returned to the caller |

Validation is handled by a Pydantic model, so malformed input is rejected at the boundary with a
descriptive `422` before it ever reaches the estimator. Error details are deliberately not echoed back
on a `500`, to avoid leaking internals to callers.

---

## The model

A `RandomForestClassifier` of **15 trees at `max_depth=5`**, trained on the 30 features of the ULB
credit-card fraud dataset: `Time`, `V1`–`V28`, `Amount`.

Two properties of this binary drive design decisions here:

**`V1`–`V28` are principal components.** The dataset's publishers ran PCA over the original transaction
attributes and released only the components, because the underlying fields are commercially and
personally sensitive. The rotation matrix was never published. This service therefore performs **no
feature engineering** — it expects vectors that have already been transformed. See
[Feature computation and serving](#feature-computation-and-serving) for how that stage would work in a
real deployment.

**The estimator carries no `feature_names_in_`.** It was fitted on a raw array rather than a DataFrame,
so it has no record of its own column names and will silently accept mis-ordered input, returning a
confident and wrong answer. The `FEATURE_COLUMNS` constant in `app/main.py` is the single source of
truth for ordering, and the Pydantic schema is what guarantees every field is present. Nothing
downstream will catch a mistake here, which is why the contract is pinned in one place and the row is
constructed from it explicitly.

`Time` deserves a note: in the source dataset it is *seconds elapsed since the first transaction in the
dataset* — a windowing artifact, not a property of a transaction. A production caller has no meaningful
value to supply. It is retained because the estimator requires exactly 30 columns, but inspection of
the fitted trees shows it is used in only **1 of 321 split nodes**, so it is very nearly inert. The
correct long-term fix is to retrain on 29 features rather than to keep asking clients for a number that
means nothing to them.

---

## Performance

Target is a p99 under 200ms. The model itself is nowhere near the bottleneck.

**Where the time actually goes.** Fifteen depth-5 trees is at most 75 comparisons per prediction —
single-digit microseconds. Essentially all of the request budget is JSON parsing, Pydantic validation,
row construction and the HTTP round trip. Optimisation effort belongs in the request path, not the
estimator.

**The endpoint is `def`, not `async def`.** This is intentional. `predict_proba` is blocking CPU-bound
work; declaring it `async` would run it directly on the event loop and stall every other in-flight
request behind it. As a sync endpoint, FastAPI dispatches it to a worker threadpool, keeping the loop
free. sklearn's tree traversal releases the GIL in its Cython inner loop, so those threads genuinely
run in parallel.

**Model loaded once at import.** Deserialising the pickle per request would dominate the latency
budget entirely. It is loaded at module import and held in process memory for the lifetime of the
container.

**Applied but worth stating explicitly:**

- Pin `n_jobs=1` on the estimator. Joblib's dispatch overhead exceeds the cost of scoring 15 tiny trees;
  parallelism here is net-negative.
- Set `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` in the container. Otherwise each uvicorn worker spawns
  a full BLAS thread pool and they oversubscribe the CPU, which shows up as erratic tail latency.
- Run `workers = CPU cores`, and scale horizontally. The workload is embarrassingly parallel and
  stateless.
- Warm the model with a dummy prediction at startup. The first call through sklearn's code paths pays
  one-off import and allocation costs, and without a warm-up that lands on a real customer request
  right after every deploy.

**Known inefficiency.** Building a one-row `pandas.DataFrame` per request costs on the order of
100–200µs — an order of magnitude more than the prediction it feeds. A plain 2-D NumPy array built
directly from `FEATURE_COLUMNS` would remove that, at the cost of some readability. It is well within
budget at current scale, but it is the first thing I would change if the p99 tightened.

---

## Deployment

**Package as an immutable image.** A multi-stage `Dockerfile` on `python:3.10-slim`: build stage
resolves Poetry dependencies into a virtualenv, runtime stage copies only that venv and the application
code, and runs as a non-root user. This keeps the final image small and reduces attack surface.

**Bake the model into the image** rather than fetching it from S3 at startup. It makes the image a
single versioned artifact — image tag `v1.4.2-model-a3f9c1` fully determines behaviour, rollback is a
tag change, and there is no network dependency in the startup path or risk of two replicas serving
different model versions after a partial restart. The trade-off is that shipping a new model means
rebuilding the image; for a model that retrains weekly or slower this is the right side of the trade.
If retraining cadence were hourly, I would switch to S3-at-startup with a checksum-verified, pinned
model version and a readiness probe that fails closed if the fetch fails.

**Runtime: ECS Fargate behind an Application Load Balancer.** The workload is stateless, CPU-bound and
horizontally scalable, so there is nothing here that justifies managing nodes. Fargate removes the
node-patching burden entirely. EKS would be the alternative if the wider platform were already
Kubernetes and shared tooling mattered more than operational simplicity.

**Services:**

| Concern | Service |
|---|---|
| Image registry | ECR, with scan-on-push |
| Compute | ECS Fargate |
| Ingress | ALB, TLS terminated at the load balancer |
| Autoscaling | Application Auto Scaling, target-tracking on request count per target |
| Config / secrets | SSM Parameter Store, Secrets Manager for anything sensitive |
| Model artifacts | S3, fronted by SageMaker Model Registry or MLflow for lineage |
| Logs / metrics / alarms | CloudWatch, with Managed Prometheus + Grafana if metrics volume justifies it |
| Tracing | AWS X-Ray or OpenTelemetry → any OTLP backend |
| CI/CD | GitHub Actions → ECR → ECS rolling or blue/green deploy via CodeDeploy |

**Rollout.** Blue/green with automatic rollback on a 5xx or p99 latency alarm. For model changes
specifically, run the new version in **shadow mode** first — mirror live traffic to it, log its scores,
compare the distribution against the incumbent, and only promote once the score distribution and
business metrics look sane on real traffic. Fraud models can pass every offline metric and still shift
approval rates in production.

**Multi-AZ, minimum two tasks.** Payment authorisation is in the critical path of taking money; a
single-task deployment makes every deploy an outage.

---

## Monitoring

Three layers, because a fraud service can be perfectly healthy by infrastructure standards while
quietly making bad decisions.

**Service health (RED).** Request rate, error rate split by status class, and duration as p50/p95/p99 —
never the mean, which hides exactly the tail the SLO is written against. Plus container CPU and memory,
ALB 5xx and target connection errors. `prometheus-fastapi-instrumentator` gives most of this from the
FastAPI side with a few lines. Alert on p99 breaching 200ms and on any sustained 5xx rate.

Structured JSON logs with a correlation ID per request, shipped to CloudWatch Logs. **No PII and no
raw feature values in logs** — see [security](#other-considerations).

**Model health.** This is what distinguishes an ML service from a normal one, and it needs to be in
place from day one:

- *Score distribution drift.* Track the daily distribution of `fraud-score` against a baseline captured
  at training time, using PSI or KL divergence. This is the leading indicator — it moves before you
  have any labels, and it catches upstream feature-pipeline breakage faster than anything else.
- *Feature drift.* Per-feature distribution monitoring against training baselines, plus null rates and
  out-of-range counts. A feature that silently starts arriving as zeros is a common and expensive
  failure.
- *Prediction volume* by segment, to catch traffic that stops arriving as well as traffic that spikes.

**Business outcomes, on a lag.** True fraud labels arrive as chargebacks, weeks to months after the
transaction. Precision, recall and AUC at the operating threshold therefore have to be computed on a
delayed join against settled outcomes and backfilled to the original prediction date — a scheduled
batch job, not a live dashboard. Alongside them, track approval rate and false-positive rate, since a
model that fixes fraud by declining good customers is a worse outcome than the fraud was.

---

## Feature computation and serving

The assignment supplies precomputed features. In production they would not be, and this is where most
of the real engineering lives.

**Split features by how fresh they need to be.** Not everything needs the same pipeline, and treating
it uniformly is how you end up paying streaming costs for data that changes monthly.

| Class | Examples | Computed by | Latency |
|---|---|---|---|
| Realtime | Transactions on this card in the last 60s / 5m / 1h; amount vs. rolling mean | Kafka or Kinesis → Flink windowed aggregations | seconds |
| Near-realtime | 24h/7d velocity, distinct-merchant counts, device history | Micro-batch Spark, every few minutes | minutes |
| Batch | Cardholder tenure, merchant risk scores, historical chargeback rate | Nightly Spark/dbt over the warehouse | hours |
| Request-time | Amount, currency, MCC, country, entry mode | Passed in the request itself | none |

**Two-tier feature store.** An offline store (S3/Parquet, queried via Athena or Spark) holding full
history for training and backfills; an online store (DynamoDB or ElastiCache/Redis) holding only the
current value per entity key, optimised for single-digit-millisecond point reads. Feast, Tecton or
SageMaker Feature Store all provide this shape; the important part is that **both tiers are populated
by the same transformation code**. Reimplementing a feature separately for training and serving is the
single most common source of train/serve skew, and it produces models that validate beautifully and
underperform in production.

**Point-in-time correctness in training.** Training sets must be built with as-of joins that reconstruct
each feature's value *as it was at the moment of the transaction*, never its current value. Getting this
wrong leaks the future into the training set and inflates offline metrics on a model that cannot
reproduce them live.

**Access at inference.** The request carries identifiers — `transaction_id`, `card_id`, `merchant_id` —
and the service does one batched key lookup (`BatchGetItem` / Redis `MGET`) against the online store,
merges the result with the request-time fields, and scores. Within a 200ms p99 budget: ~10–20ms for the
feature fetch, single-digit ms for inference, the rest headroom for network and serialisation.

**Fail soft, never block the payment.** If the online store times out or misses, fall back to training
medians or a conservative default, flag the response as degraded, and emit a metric on it. A fraud
service that returns nothing forces the caller to choose between declining every transaction or
approving every transaction — both are worse than a slightly less accurate score. The degraded-response
rate then becomes a first-class alert.

**The PCA caveat.** For this specific model, `V1`–`V28` come from a PCA rotation that was never
published, so no feature pipeline could reproduce them. In a real system the fitted transformer would be
bundled with the estimator in a single `sklearn.Pipeline` and versioned as one artifact, so that
preprocessing can never drift out of sync with the model that depends on it.

---

## Other considerations

**Return the model version.** Adding `model_version` to the response payload costs nothing and makes
post-hoc analysis possible — without it, a scored transaction cannot be attributed to the model that
scored it, which makes A/B tests and incident forensics guesswork.

**Echo a correlation ID.** Accepting and returning `transaction_id` lets predictions be joined to
outcomes later, which is a hard prerequisite for the delayed-label monitoring described above.

**Pickle deserialisation is arbitrary code execution.** `pickle.load` will execute whatever is in the
file. Acceptable here because the artifact is trusted and baked into the image, but the model path must
never be attacker-controllable, and in a real pipeline the artifact should be checksum-verified against
the model registry before loading. ONNX or skops would remove the risk class entirely.

**Data handling.** Transaction features are personal data under GDPR. No feature values or identifiers
in application logs; sampled request/response payloads for drift analysis should go to a separate,
access-controlled store with an explicit retention policy, not into general logging infrastructure.

**Network posture.** This should not be internet-facing. Deploy into a private subnet, expose it only to
the payment authorisation service via an internal ALB with security-group restrictions, and authenticate
callers with mTLS or a gateway-issued token.

**Health and readiness must be distinct.** Liveness answers "is the process up"; readiness must confirm
the model is loaded *and* that a warm prediction succeeds. Otherwise the load balancer routes traffic to
containers that are running but cannot score, and the failure surfaces as 500s rather than as a failed
deploy.

**Explainability.** A fraud decision that cannot be explained is a problem for chargeback teams,
customer support and regulators. Because `V1`–`V28` are anonymised components, no reason code derived
from them is human-meaningful — "declined because V14 was low" is not something you can tell anyone. A
production system needs features that map back to explainable concepts, which is a modelling
constraint, not something the serving layer can fix.

---

## Testing

`pytest` with FastAPI's `TestClient`, which drives the ASGI app in-process — no server, no sockets, so
the suite runs in well under a second. 25 tests in four groups:

**Contract.** The load-bearing group. Because the estimator has no `feature_names_in_`, mis-ordered
input produces a confident wrong answer that nothing at runtime would flag. These assert
`FEATURE_COLUMNS == list(Features.model_fields)`, that the count matches `model.n_features_in_`, that
the constructed row carries values in declared order, and that `classes_ == [0, 1]` — without which
`predict_proba[:, 1]` would be the *non*-fraud probability.

**Scoring.** Status and range, agreement with a direct `predict_proba` call, determinism, and
independence from JSON key order. Plus a sensitivity sweep over `V14`, `V17` and `V12` — the most
heavily split features in the forest — asserting the score actually moves. Nearly every transaction
scores close to zero, so a bug returning a constant would otherwise satisfy every other assertion here.

**Validation.** Missing, non-numeric, null and unknown fields all `422`; integers coerce to floats;
`GET` is `405`.

**Failure handling.** A monkeypatched estimator that raises must yield a `500` whose body contains the
generic message and none of the exception text.

Not yet covered: a load test (Locust or `k6`) to substantiate the p99 target rather than reason about
it, and a golden-value regression test pinning the exact score for the mock body, which would catch an
unintended model swap.

---

## Given more time

Roughly in the order I would tackle them:

1. **Startup lifecycle.** Move model loading into a FastAPI `lifespan` handler with a warm-up
   prediction, resolve `MODEL_PATH` relative to the package root rather than the working directory, and
   fail fast with a clear message if the artifact is missing. Import-time loading makes the module
   CWD-sensitive to run and awkward to test — `tests/conftest.py` currently has to set `MODEL_PATH` to
   an absolute path *before* importing the app, which is a workaround for exactly this.
2. **`/health` and `/ready` endpoints.** Readiness should run a real prediction, not just confirm the
   process is up. The container healthcheck currently probes `/openapi.json`, which proves only that
   HTTP is being served.
3. **Observability**: `model_version` and `transaction_id` on the response schema, structured JSON
   logging with a correlation ID, and Prometheus instrumentation.
4. **Replace the per-request DataFrame** with direct NumPy array construction. This also silences a
   `UserWarning` sklearn emits on *every* prediction — "X has feature names, but
   RandomForestClassifier was fitted without feature names" — which at production request rates is
   meaningful log volume for no information.
5. **Load testing** to verify the p99 target under realistic concurrency.
6. **Retrain without `Time`**, or wrap the estimator in a `Pipeline`, so the API contract stops carrying
   a field that means nothing to callers.
7. **`docker-compose.yml`** for a one-command local stack once there are dependencies (feature store,
   metrics collector) to bring up alongside the service.


![Alt text](app/assets/image.png "Post Man for Testing API response time")