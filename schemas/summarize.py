from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SummaryLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=50,
        max_length=50_000,
        description="The text to summarize.",
    )
    summary_length: SummaryLength = Field(
        default=SummaryLength.medium,
        description="Desired summary length: short, medium, or long.",
    )
    extract_keywords: bool = Field(
        default=False,
        description="Whether to extract keywords from the text.",
    )


class SummarizeResponse(BaseModel):
    summary: str
    length: SummaryLength
    status: str = "success"
    char_count_original: int
    char_count_summary: int
    keywords: Optional[list[str]] = None
    cached: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    cache_enabled: bool
