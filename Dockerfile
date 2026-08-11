# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: resolve dependencies into a self-contained virtualenv.
#
# Poetry and its transitive dependencies stay in this stage and never reach the
# runtime image. Python is pinned to 3.10 because scikit-learn 1.0.2 — required
# to unpickle the model binary — has no wheels for 3.11+.
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Matches the version that generated poetry.lock (see its header), so the
    # lock format is read by the tool that wrote it.
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

# Copied ahead of the application code so this layer is cached and dependency
# resolution only re-runs when the manifests themselves change.
COPY pyproject.toml poetry.lock ./

# --only main excludes dev tooling. The lock file makes this build reproducible;
# without it, a transitive release could silently break sklearn 1.0.2 compat.
RUN poetry install --only main --no-root && rm -rf "${POETRY_CACHE_DIR}"


# ---------------------------------------------------------------------------
# Runtime stage: virtualenv + application code only.
# ---------------------------------------------------------------------------
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    # Absolute so the service is independent of the working directory it is
    # started from. The model is baked into the image, making the image tag a
    # complete description of serving behaviour — see README, "Deployment".
    MODEL_PATH=/app/models/fraud-model.pickle \
    # Each uvicorn worker would otherwise spawn a full BLAS thread pool; on a
    # multi-worker container those pools oversubscribe the CPU and show up as
    # erratic tail latency. The model is 15 depth-5 trees, so intra-op
    # parallelism buys nothing here anyway.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    # One process per container; scale by adding tasks rather than workers, so
    # autoscaling and per-replica metrics stay meaningful. Override at deploy
    # time if you would rather saturate a larger task.
    WEB_CONCURRENCY=1

WORKDIR /app

# Unprivileged, no shell, no home directory — nothing in this image needs them.
RUN groupadd --system --gid 10001 sirius \
    && useradd --system --uid 10001 --gid sirius --no-create-home --shell /usr/sbin/nologin sirius

COPY --from=builder --chown=sirius:sirius /app/.venv /app/.venv

# Model before application code: it changes far less often, so it stays cached
# across ordinary code edits.
COPY --chown=sirius:sirius models/ ./models/
COPY --chown=sirius:sirius app/ ./app/

USER sirius

EXPOSE 8888

# curl is deliberately absent from the runtime image, so the probe uses the
# interpreter that is already here. This is liveness only: /health answers "is
# the process up", which is the question a container restart can fix. Whether
# the model can score is /ready, and that belongs on the ECS/ALB target group,
# which can drain a task instead of killing it.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8888/health', timeout=2)" \
    || exit 1

# Exec form, so uvicorn is PID 1 and receives SIGTERM directly. Under the shell
# form it would be orphaned behind /bin/sh and killed only after the container
# timeout, cutting off in-flight requests on every deploy.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
