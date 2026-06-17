from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Clinical question about ambulance protocols")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Answer grounded in the ambulance protocol documents")
