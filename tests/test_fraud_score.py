"""Tests for the ``POST /fraud-score`` endpoint.

Grouped into four concerns:

* **Contract**  — invariants that nothing at runtime would otherwise catch,
  most importantly feature ordering.
* **Scoring**   — the happy path, and agreement with the estimator itself.
* **Validation** — malformed input is rejected at the boundary.
* **Failure**   — inference errors surface as 500 without leaking internals.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import FEATURE_COLUMNS, Features

ENDPOINT = "/fraud-score"
# Wire format per the brief: hyphenated, not the Python attribute name.
SCORE_KEY = "fraud-score"


class TestContract:
    """Guards on the request/model contract.

    The estimator was fitted on a bare array and carries no ``feature_names_in_``,
    so it will accept mis-ordered input and return a confident, wrong answer.
    Nothing downstream detects that — these tests are the only line of defence.
    """

    def test_feature_columns_matches_schema_order(self) -> None:
        assert FEATURE_COLUMNS == list(Features.model_fields)

    def test_feature_count_matches_estimator(self, model) -> None:
        assert len(FEATURE_COLUMNS) == model.n_features_in_ == 30

    def test_mock_body_covers_exactly_the_schema(self, mock_body) -> None:
        assert set(mock_body) == set(FEATURE_COLUMNS)

    def test_row_is_built_in_declared_order(self, body) -> None:
        """The DataFrame handed to the estimator must follow FEATURE_COLUMNS."""
        row = pd.DataFrame([Features(**body).model_dump()], columns=FEATURE_COLUMNS)

        assert list(row.columns) == FEATURE_COLUMNS
        assert row.iloc[0].tolist() == [body[column] for column in FEATURE_COLUMNS]

    def test_estimator_is_binary_with_fraud_as_class_one(self, model) -> None:
        """`predict_proba[:, 1]` is only the fraud probability if class 1 is fraud."""
        assert list(model.classes_) == [0, 1]


class TestScoring:
    def test_returns_a_probability(self, client: TestClient, body) -> None:
        response = client.post(ENDPOINT, json=body)

        assert response.status_code == 200
        score = response.json()[SCORE_KEY]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_agrees_with_the_estimator(self, client: TestClient, body, model) -> None:
        """The HTTP layer must not distort the model's own output."""
        row = pd.DataFrame([body], columns=FEATURE_COLUMNS)
        expected = model.predict_proba(row)[0, 1]

        response = client.post(ENDPOINT, json=body)

        assert response.json()[SCORE_KEY] == pytest.approx(expected)

    def test_is_deterministic(self, client: TestClient, body) -> None:
        first = client.post(ENDPOINT, json=body).json()
        second = client.post(ENDPOINT, json=body).json()

        assert first == second

    def test_json_key_order_does_not_affect_the_score(
        self, client: TestClient, body
    ) -> None:
        """Scoring must depend on field names, not on JSON key order.

        A regression here would mean the row is being built from insertion order
        rather than from FEATURE_COLUMNS.
        """
        reversed_body = {key: body[key] for key in reversed(list(body))}

        baseline = client.post(ENDPOINT, json=body).json()[SCORE_KEY]
        shuffled = client.post(ENDPOINT, json=reversed_body).json()[SCORE_KEY]

        assert shuffled == baseline

    @pytest.mark.parametrize("feature", ["V14", "V17", "V12"])
    def test_score_responds_to_input(
        self, client: TestClient, body, feature: str
    ) -> None:
        """Sweeping a heavily-split feature must move the score.

        Most transactions score at or near zero, so a bug that returned a constant
        would satisfy every other assertion in this file. These three features
        account for the largest share of split nodes in the fitted forest, so a
        wide sweep is guaranteed to cross thresholds.
        """
        scores = set()
        for value in (-20.0, -5.0, 0.0, 5.0, 20.0):
            scores.add(
                client.post(ENDPOINT, json={**body, feature: value}).json()[SCORE_KEY]
            )

        assert len(scores) > 1, f"score is insensitive to {feature}"


class TestValidation:
    @pytest.mark.parametrize("missing", ["Time", "V1", "V14", "Amount"])
    def test_missing_field_is_rejected(
        self, client: TestClient, body, missing: str
    ) -> None:
        del body[missing]

        response = client.post(ENDPOINT, json=body)

        assert response.status_code == 422
        assert missing in str(response.json())

    @pytest.mark.parametrize("bad_value", ["not-a-number", None, [], {}])
    def test_non_numeric_value_is_rejected(
        self, client: TestClient, body, bad_value
    ) -> None:
        body["V1"] = bad_value

        assert client.post(ENDPOINT, json=body).status_code == 422

    def test_unknown_field_is_rejected(self, client: TestClient, body) -> None:
        """Strict schema: an unrecognised field is a client bug, not something to ignore."""
        body["V29"] = 1.0

        assert client.post(ENDPOINT, json=body).status_code == 422

    def test_empty_body_is_rejected(self, client: TestClient) -> None:
        assert client.post(ENDPOINT, json={}).status_code == 422

    def test_integers_are_accepted_as_floats(self, client: TestClient, body) -> None:
        """`Time: 0` in the mock body is an int; coercion must not be a 422."""
        body["Time"] = 0
        body["Amount"] = 150

        assert client.post(ENDPOINT, json=body).status_code == 200

    def test_get_is_not_allowed(self, client: TestClient) -> None:
        assert client.get(ENDPOINT).status_code == 405


class TestFailureHandling:
    def test_inference_error_returns_500_without_leaking_details(
        self, client: TestClient, body, monkeypatch
    ) -> None:
        from app import main

        secret = "connection string leaked in exception text"

        class BrokenModel:
            def predict_proba(self, _row):
                raise RuntimeError(secret)

        monkeypatch.setattr(main, "model", BrokenModel())

        response = client.post(ENDPOINT, json=body)

        assert response.status_code == 500
        assert response.json()["detail"] == "Unable to generate fraud score"
        assert secret not in response.text
