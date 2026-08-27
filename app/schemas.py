"""Pydantic request/response models for the inference API.

The request model mirrors the ORIGINAL untransformed customer fields expected
by the trained model. Users do not need to provide one-hot encoded features.
"""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """A single customer using the original (untransformed) feature names."""

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class ModelInfoResponse(BaseModel):
    model: str
    threshold: float
    feature_count: int


class HealthResponse(BaseModel):
    status: str