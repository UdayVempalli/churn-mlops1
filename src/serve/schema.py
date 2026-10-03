"""Request and response models for the prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class Customer(BaseModel):
    """One customer record, in the raw dataset's shape.

    Raw values are fine because encoding lives inside the model
    pipeline, not in a separate step.
    """

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn: int
    probability: float
    threshold: float


class Health(BaseModel):
    status: str
    model_loaded: bool