from typing import Optional
from pydantic import BaseModel, Field


# --- Agent structured output ---

class ProtocolsOutput(BaseModel):
    protocols: list[str] = Field(
        ...,
        description="Ordered list of prehospital/dispatch protocols relevant to the incident. Each item is one actionable step."
    )


# --- Dispatch request payload ---

class PatientInfo(BaseModel):
    ageGroup: Optional[str] = None
    approxAge: Optional[str] = None
    sex: Optional[str] = None

    model_config = {"extra": "ignore"}


class IncidentInfo(BaseModel):
    description: str = Field(..., description="Free-text incident description — primary RAG input")
    priority: Optional[str] = None
    consciousness: Optional[str] = None
    breathing: Optional[str] = None
    activelyBleeding: Optional[bool] = None
    medicalHistory: Optional[str] = None

    model_config = {"extra": "ignore"}


class DispatchRequest(BaseModel):
    dispatchId: str
    patientInfo: Optional[PatientInfo] = None
    incidentInfo: IncidentInfo

    model_config = {"extra": "ignore"}


# --- API response ---

class ProtocolResponse(BaseModel):
    dispatchId: str
    incidentDescription: str
    protocols: list[str] = Field(..., description="Ordered list of actionable prehospital protocols")
