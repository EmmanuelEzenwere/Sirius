"""Shared fixtures for the Sirius test suite.

``app.main`` loads the model at import time from a path that is relative to the
working directory, so ``MODEL_PATH`` is resolved to an absolute location *before*
the application is imported. Without this the suite would only pass when pytest
happened to be invoked from the repository root.
"""

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_REQUEST_BODY = Path(__file__).resolve().parent / "mock-request-body.json"

# setdefault, not assignment, so an explicit MODEL_PATH in the environment still
# wins — useful for running the suite against a candidate model binary.
os.environ.setdefault("MODEL_PATH", str(REPO_ROOT / "models" / "fraud-model.pickle"))

from app.main import FEATURE_COLUMNS, Features, app  # noqa: E402  (import after env setup)
from app import main  # noqa: E402


@pytest.fixture(scope="session")
def mock_body() -> dict[str, float]:
    """The supplied mock request body, as shipped."""
    with MOCK_REQUEST_BODY.open() as handle:
        return json.load(handle)


@pytest.fixture
def body(mock_body: dict[str, float]) -> dict[str, float]:
    """A per-test mutable copy, so mutation cannot leak between tests."""
    return dict(mock_body)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def model() -> Any:
    """The loaded estimator, for asserting the API agrees with it directly."""
    return main.model


__all__ = ["FEATURE_COLUMNS", "Features", "app", "main"]
