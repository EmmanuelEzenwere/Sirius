"""Shared fixtures for the Sirius test suite.

``app.main`` resolves its default ``MODEL_PATH`` relative to the package root,
so the app imports and loads the model regardless of the directory pytest is
invoked from. Set ``MODEL_PATH`` explicitly to run the suite against a different
model binary.

The model is loaded by the app's ``lifespan`` handler, so importing ``app.main``
has no side effects; ``main.model`` is only populated once the ``client`` fixture
has entered the ``TestClient`` context and triggered startup.
"""

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import FEATURE_COLUMNS, Features, app
from app import main

MOCK_REQUEST_BODY = Path(__file__).resolve().parent / "mock-request-body.json"


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
    # The context manager runs the lifespan handler, which loads and warms the
    # model, so main.model is populated for the duration of the session.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def model(client: TestClient) -> Any:
    """The loaded estimator, for asserting the API agrees with it directly.

    Depends on ``client`` so lifespan startup has populated ``main.model`` before
    it is read.
    """
    return main.model


__all__ = ["FEATURE_COLUMNS", "Features", "app", "main"]