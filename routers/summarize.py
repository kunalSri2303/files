import time
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from google.api_core.exceptions import (
    ResourceExhausted,
    DeadlineExceeded,
    PermissionDenied,
    InvalidArgument,
    GoogleAPIError,
)

from schemas.summarize import SummarizeRequest, SummarizeResponse, ErrorResponse
from services.summarizer import summarization_service
from utils.rate_limiter import rate_limit_dependency
from utils.logger import request_logger

router = APIRouter(prefix="/summarize", tags=["Summarization"])


@router.post(
    "",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Gemini or server error"},
        503: {"model": ErrorResponse, "description": "Gemini service unavailable"},
    },
    summary="Summarize text using Gemini",
    description=(
        "Accepts a block of text and returns a concise AI-generated summary via Google Gemini. "
        "Optionally extracts keywords and controls summary length."
    ),
)
async def summarize_text(
    request_body: SummarizeRequest,
    http_request: Request,
    _rate_limited: None = Depends(rate_limit_dependency),
) -> SummarizeResponse:
    start = time.perf_counter()
    client_ip = http_request.client.host if http_request.client else "unknown"

    request_logger.info(
        "POST /summarize",
        extra={
            "client_ip": client_ip,
            "input_chars": len(request_body.text),
            "summary_length": request_body.summary_length,
            "extract_keywords": request_body.extract_keywords,
        },
    )

    try:
        result = await summarization_service.summarize(request_body)

    except PermissionDenied:
        request_logger.error("Gemini authentication failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="Gemini authentication failed",
                detail="The GEMINI_API_KEY is invalid or missing. Check your .env file.",
                code=503,
            ).model_dump(),
        )

    except ResourceExhausted:
        request_logger.warning("Gemini rate limit / quota exceeded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="Gemini quota exceeded",
                detail="The Gemini API quota was reached. Please try again shortly.",
                code=503,
            ).model_dump(),
        )

    except DeadlineExceeded:
        request_logger.error("Gemini request timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=ErrorResponse(
                error="Gemini request timed out",
                detail="Gemini did not respond in time. Please retry.",
                code=504,
            ).model_dump(),
        )

    except InvalidArgument as e:
        request_logger.error("Gemini invalid argument", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="Invalid request to Gemini",
                detail=str(e),
                code=400,
            ).model_dump(),
        )

    except GoogleAPIError as e:
        request_logger.error("Gemini API error", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=ErrorResponse(
                error="Gemini API error",
                detail=str(e),
                code=502,
            ).model_dump(),
        )

    except Exception as e:
        request_logger.error("Unexpected error during summarization", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="Internal server error",
                detail="An unexpected error occurred. Please try again.",
                code=500,
            ).model_dump(),
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    request_logger.info(
        "POST /summarize completed",
        extra={
            "elapsed_ms": elapsed_ms,
            "cached": result.cached,
            "output_chars": result.char_count_summary,
        },
    )

    return result