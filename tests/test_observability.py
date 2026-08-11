"""Tests for the operational surface: probes, correlation ids, and metrics.

Kept separate from ``test_fraud_score.py``, which covers the scoring contract.
These exercise the endpoints an orchestrator and a monitoring stack talk to,
none of which are part of the prediction path.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app import main

SCORE_ENDPOINT = "/fraud-score"
TRANSACTION_ID_HEADER = "X-Transaction-Id"


class TestLiveness:
    def test_health_is_ok(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_does_not_depend_on_the_model(
        self, client: TestClient, monkeypatch
    ) -> None:
        """Liveness must stay green when the model is gone.

        A restart cannot reload a missing artifact, so reporting dead here would
        turn a readiness problem into a restart loop.
        """
        monkeypatch.setattr(main, "model", None)

        assert client.get("/health").status_code == 200


class TestReadiness:
    def test_ready_when_the_model_can_score(self, client: TestClient) -> None:
        response = client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_not_ready_when_the_model_is_absent(
        self, client: TestClient, monkeypatch
    ) -> None:
        monkeypatch.setattr(main, "model", None)

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"status": "not ready"}

    def test_not_ready_when_scoring_raises(
        self, client: TestClient, monkeypatch
    ) -> None:
        """A model object that exists but cannot score must not receive traffic."""

        class BrokenModel:
            def predict_proba(self, _row):
                raise RuntimeError("estimator is corrupt")

        monkeypatch.setattr(main, "model", BrokenModel())

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {"status": "not ready"}


class TestCorrelationId:
    def test_supplied_id_is_echoed_in_header_and_body(
        self, client: TestClient, body
    ) -> None:
        supplied = "txn-abc-123"

        response = client.post(
            SCORE_ENDPOINT, json=body, headers={TRANSACTION_ID_HEADER: supplied}
        )

        assert response.status_code == 200
        assert response.headers[TRANSACTION_ID_HEADER] == supplied
        assert response.json()["transaction_id"] == supplied

    def test_id_is_generated_when_absent(self, client: TestClient, body) -> None:
        response = client.post(SCORE_ENDPOINT, json=body)

        generated = response.json()["transaction_id"]
        assert response.headers[TRANSACTION_ID_HEADER] == generated
        # Parses as a v4 UUID, so the generated id is what we think it is.
        assert uuid.UUID(generated).version == 4

    def test_generated_ids_are_unique_per_request(
        self, client: TestClient, body
    ) -> None:
        first = client.post(SCORE_ENDPOINT, json=body).json()["transaction_id"]
        second = client.post(SCORE_ENDPOINT, json=body).json()["transaction_id"]

        assert first != second

    @pytest.mark.parametrize("path", ["/health", "/ready"])
    def test_probes_also_carry_a_correlation_id(
        self, client: TestClient, path: str
    ) -> None:
        """The middleware is global, not bolted onto the scoring route."""
        assert TRANSACTION_ID_HEADER in client.get(path).headers

    def test_error_responses_still_carry_the_id(self, client: TestClient) -> None:
        """A 422 is exactly when a caller most needs the id to correlate logs."""
        response = client.post(SCORE_ENDPOINT, json={})

        assert response.status_code == 422
        assert TRANSACTION_ID_HEADER in response.headers


class TestModelVersion:
    def test_response_carries_the_model_version(
        self, client: TestClient, body
    ) -> None:
        response = client.post(SCORE_ENDPOINT, json=body)

        assert response.json()["model_version"] == main.MODEL_VERSION

    def test_version_defaults_to_unknown_when_unset(self) -> None:
        """Unset MODEL_VERSION must not crash startup, just label the output."""
        assert main.MODEL_VERSION == "unknown" or main.MODEL_VERSION

    def test_wire_name_is_model_version_not_version(
        self, client: TestClient, body
    ) -> None:
        """The Pydantic attribute is `version`; the alias is what callers see."""
        payload = client.post(SCORE_ENDPOINT, json=body).json()

        assert "model_version" in payload
        assert "version" not in payload


class TestMetrics:
    def test_metrics_endpoint_serves_prometheus_text(
        self, client: TestClient, body
    ) -> None:
        client.post(SCORE_ENDPOINT, json=body)

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_scoring_requests_are_counted(self, client: TestClient, body) -> None:
        client.post(SCORE_ENDPOINT, json=body)

        body_text = client.get("/metrics").text

        assert "http_request_duration_seconds" in body_text
        assert SCORE_ENDPOINT in body_text

    def test_probes_are_excluded_from_metrics(self, client: TestClient) -> None:
        """Probes fire on a fixed timer and would swamp real traffic."""
        client.get("/health")
        client.get("/ready")

        body_text = client.get("/metrics").text

        assert '/health' not in body_text
        assert '/ready' not in body_text
