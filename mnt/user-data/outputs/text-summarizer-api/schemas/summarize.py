from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional, List


class SummaryLength(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"


class SummarizeRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text to summarize",
        examples=["Long article or document text goes here..."],
    )
    summary_length: SummaryLength = Field(
        default=SummaryLength.medium,
        description="Desired summary length: short, medium, or long",
    )
    extract_keywords: bool = Field(
        default=False,
        description="Whether to extract key topics from the text",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty or whitespace only.")
        if len(v) < 50:
            raise ValueError("Text must be at least 50 characters long to summarize.")
        if len(v) > 50_000:
            raise ValueError(
                f"Text exceeds maximum allowed length of 50,000 characters "
                f"(received {len(v):,} characters)."
            )
        return v


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="The generated summary")
    length: SummaryLength = Field(..., description="The summary length used")
    status: str = Field(default="success", description="Response status")
    char_count_original: int = Field(..., description="Character count of original text")
    char_count_summary: int = Field(..., description="Character count of summary")
    keywords: Optional[List[str]] = Field(
        default=None, description="Extracted keywords if requested"
    )
    cached: bool = Field(default=False, description="Whether result was served from cache")


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    cache_enabled: bool


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None
    code: int
