"""Sirius fraud-scoring API.

Serves a pre-trained scikit-learn classifier over HTTP. A single POST to
``/fraud-score`` returns the model's estimated probability that a transaction
is fraudulent, so that callers can apply their own decision threshold rather
than receiving a hard accept/decline.

Inputs are the 30 features of the ULB credit-card fraud dataset: ``Time``,
``V1``-``V28`` and ``Amount``. ``V1``-``V28`` are anonymised principal
components, so this service expects vectors that have *already* been through
the dataset's PCA transform -- it does no feature engineering of its own.
``FEATURE_COLUMNS`` pins the column order the model was fitted on; the
estimator carries no ``feature_names_in_``, so ordering is enforced here or
not at all.

The model is loaded once at import time from ``MODEL_PATH`` (see ``.env``,
defaulting to ``models/fraud-model.pickle``) and held in module state for the
process lifetime.
"""


import logging
import os

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from .env file

app = FastAPI()

class Features(BaseModel):  # pylint: disable=too-few-public-methods
    """Transaction features for fraud score prediction.

    Field declaration order is significant: FEATURE_COLUMNS is asserted against
    it in the test suite, and the estimator has no feature names of its own.
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
    fraud_score: float = Field(serialization_alias="fraud-score")

FEATURE_COLUMNS = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    'V6',
    'V7',
    'V8',
    'V9',
    'V10',
    'V11',
    'V12',
    'V13',
    'V14',
    'V15',
    'V16',
    'V17',
    'V18',
    'V19',
    'V20',
    'V21',
    'V22',
    'V23',
    'V24',
    'V25',
    'V26',
    'V27',
    'V28',
    'Amount']

# Load Fraud Prevention model
model_path = os.getenv("MODEL_PATH", "models/fraud-model.pickle")
model = load_model(model_path=model_path)


@app.post("/fraud-score", response_model=FraudResponse)
def predict_fraud_score(data: Features) -> FraudResponse:
    """predict fraud score for a single transaction

    Args:
        data (Features): input features for the transaction

    Returns:
        FraudResponse: The fraud score for the transaction
    """
    row = pd.DataFrame([data.model_dump()], columns=FEATURE_COLUMNS)
    # fraud score is the probability of the transaction being fraudulent
    try:
        fraud_score = float(model.predict_proba(row.to_numpy())[0,1])
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate fraud score",
        ) from exc
    return FraudResponse(fraud_score=fraud_score)


if __name__ == "__main__":

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8888,
        reload=True,
    )