"""Sirius fraud-scoring API.

Serves a pre-trained scikit-learn classifier over HTTP. A single POST to
``/fraud-score`` returns the model's estimated probability that a transaction
is fraudulent, so that callers can apply their own decision threshold rather
than receiving a hard accept/decline.

Inputs are the 30 features of the ULB credit-card fraud dataset: ``Time``,
``V1`` to ``V28`` and ``Amount``. ``V1`` to ``V28`` are anonymised principal
components, so this service expects vectors that have *already* been through
the dataset's PCA transform. It does no feature engineering of its own.
``FEATURE_COLUMNS`` pins the column order the model was fitted on; the
estimator carries no ``feature_names_in_``, so ordering is enforced here or
not at all.

The model is loaded once on startup by a ``lifespan`` handler from ``MODEL_PATH``
(see ``.env``), defaulting to the bundled ``models/fraud-model.pickle`` resolved
relative to the package root, and held in module state for the process lifetime.
Loading on startup rather than at import keeps the module importable without
side effects and surfaces a missing artifact as a clear startup failure.

Alongside scoring the module exposes ``GET /health`` (liveness), ``GET /ready``
(readiness, which scores a throwaway row to prove the model works) and
``GET /metrics`` (Prometheus). Every request carries an ``X-Transaction-Id``
correlation id, supplied by the caller or generated here, echoed in the response
header and body and stamped onto every JSON log line.
"""


import contextvars
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field
from pythonjsonlogger.json import JsonFormatter
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

from .utils import load_model

load_dotenv()  # Load environment variables from .env file

# Correlation id for the request being served. A ContextVar rather than a
# parameter threaded through every call, so log records can pick it up without
# each logging site having to know about it. Sync endpoints run in a threadpool,
# and anyio copies the context into the worker thread, so this stays correct.
TRANSACTION_ID_HEADER = "X-Transaction-Id"
_transaction_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "transaction_id", default="-"
)

# Label for the artifact currently being served. Set by the deployment pipeline
# so a score can be attributed to the exact model that produced it; without it,
# A/B analysis and incident forensics are guesswork.
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")


class _TransactionIdFilter(logging.Filter):
    """Stamp every record with the current request's correlation id."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.transaction_id = _transaction_id.get()
        return True


def configure_logging() -> None:
    """Emit JSON to stdout, one object per line.

    Structured logs are what make correlation ids useful: CloudWatch Logs
    Insights and the like can filter on ``transaction_id`` as a field rather
    than regex-matching a formatted string.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s %(transaction_id)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )
    handler.addFilter(_TransactionIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


configure_logging()
logger = logging.getLogger(__name__)


class TransactionIdMiddleware:
    """Assign a correlation id to every request and echo it back.

    Written as raw ASGI rather than ``BaseHTTPMiddleware`` deliberately: the
    latter wraps each request in an anyio task group with message streams, which
    is measurable overhead on a service whose entire budget is 200ms. The
    ContextVar is set before the downstream app is awaited, so the value is
    captured by the request's context and visible to handlers and log records.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Honour a caller-supplied id so a transaction can be traced across
        # service boundaries; mint one only when the caller has not.
        incoming = Headers(scope=scope).get(TRANSACTION_ID_HEADER)
        transaction_id = incoming or str(uuid.uuid4())
        token = _transaction_id.set(transaction_id)

        async def send_with_transaction_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[TRANSACTION_ID_HEADER] = transaction_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_transaction_id)
        finally:
            _transaction_id.reset(token)


def current_transaction_id() -> str:
    """Dependency exposing the id the middleware assigned to this request.

    Reads the ContextVar rather than the header directly, so a request that
    arrived without one gets the same generated id that was echoed in the
    response header.
    """
    return _transaction_id.get()


class Features(BaseModel):  # pylint: disable=too-few-public-methods
    """Transaction features for fraud score prediction.

    Field declaration order is significant: FEATURE_COLUMNS is derived from it,
    and the estimator has no feature names of its own.
    """

    # Reject unrecognised fields rather than silently dropping them. A caller
    # sending an unexpected key has misunderstood the contract, and for a
    # payments API that is worth surfacing as a 422 rather than absorbing.
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class FraudResponse(BaseModel):
    """Scoring result for a single transaction."""

    fraud_score: float = Field(serialization_alias="fraud-score")
    # Pydantic v2 reserves the `model_` attribute prefix, so the field is named
    # `version` and the wire name is carried by the alias. The alternative,
    # disabling protected_namespaces, would silence the guard for every future
    # field on this model rather than just this one.
    version: str = Field(serialization_alias="model_version")
    transaction_id: str


# Single source of truth for column order, derived from the schema so the two
# can never drift. Pydantic v2 preserves field-declaration order, and
# model_config is a ClassVar rather than a field, so it does not appear here.
FEATURE_COLUMNS = list(Features.model_fields)

# Resolve the default model path relative to the package root (app/main.py, up
# to the repo root, then models/), so it works regardless of the directory the
# process is launched from. An explicit MODEL_PATH still overrides, e.g. to
# point at a candidate binary.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud-model.pickle"

# Populated on startup by the lifespan handler; held for the process lifetime.
model = None


def _warm_up_row() -> np.ndarray:
    """A single all-zero row of the right width, for warm-up and readiness."""
    return np.zeros((1, len(FEATURE_COLUMNS)), dtype=float)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup, fail fast if it is missing, then warm it.

    Deferring the load to startup rather than import time keeps the module
    importable without side effects. The warm-up prediction pays sklearn's
    one-off first-call costs here, at startup, instead of on the first customer
    request right after a deploy.
    """
    global model

    # uvicorn installs its own handlers when it starts, after this module was
    # imported, so its access and error logs would bypass the JSON formatter.
    # Clearing them here (startup runs after uvicorn's logging setup) lets those
    # records propagate to the root handler and come out structured too.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    if not Path(path).exists():
        raise RuntimeError(f"Model artifact not found at {path!r}")
    model = load_model(model_path=path)
    model.predict_proba(_warm_up_row())
    logger.info("Model loaded and warmed", extra={"model_path": path, "model_version": MODEL_VERSION})
    yield
    model = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(TransactionIdMiddleware)

# Request count, latency histogram and in-flight gauge on /metrics. Probe and
# scrape endpoints are excluded: they run on a fixed timer, so counting them
# would drown out real traffic and drag the latency percentiles toward the
# trivial handlers.
Instrumentator(
    should_group_status_codes=False,
    should_instrument_requests_inprogress=True,
    inprogress_labels=True,
    excluded_handlers=["/metrics", "/health", "/ready"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness probe: the process is up and serving HTTP.

    Deliberately checks nothing else. A liveness failure means "restart me", and
    restarting will not fix a broken dependency, so folding dependency checks in
    here turns a downstream problem into a restart loop.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response) -> Dict[str, str]:
    """Readiness probe: the model is loaded and can actually score.

    Runs a real prediction rather than a truthiness check on the global, because
    "the object exists" and "the object can score" are different claims, and only
    the second one means traffic should be routed here.
    """
    if model is None:
        response.status_code = 503
        return {"status": "not ready"}

    try:
        model.predict_proba(_warm_up_row())
    except Exception:
        logger.exception("Readiness probe failed")
        response.status_code = 503
        return {"status": "not ready"}

    return {"status": "ready"}


@app.post("/fraud-score", response_model=FraudResponse)
def predict_fraud_score(
    data: Features,
    transaction_id: str = Depends(current_transaction_id),
) -> FraudResponse:
    """Predict the fraud score for a single transaction.

    Args:
        data (Features): input features for the transaction
        transaction_id (str): correlation id for this request, from the
            ``X-Transaction-Id`` header or generated if absent

    Returns:
        FraudResponse: the fraud score, the model version that produced it, and
            the correlation id
    """
    # Build a one-row 2-D array in the pinned column order. Constructing the
    # array directly, rather than via a DataFrame, keeps per-request DataFrame
    # overhead off the hot path; and because the estimator was fitted without
    # feature names, a bare array also avoids sklearn's per-call
    # "X has feature names" warning.
    features = data.model_dump()
    row = np.array([[features[name] for name in FEATURE_COLUMNS]], dtype=float)
    try:
        fraud_score = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        # The correlation id is attached by the logging filter, so the failure
        # can be joined back to the caller's request without logging the body.
        # Feature values stay out of the logs: they are personal data.
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate fraud score",
        ) from exc
    return FraudResponse(
        fraud_score=fraud_score,
        version=MODEL_VERSION,
        transaction_id=transaction_id,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8888,
        reload=True,
    )