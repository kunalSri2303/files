from fastapi import APIRouter
from schemas.summarize import HealthResponse
from utils.config import get_settings
from utils.cache import summarizer_cache

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API status and basic configuration info.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        llm_provider=settings.LLM_PROVIDER,
        cache_enabled=settings.CACHE_ENABLED,
    )


@router.get(
    "/cache/stats",
    summary="Cache statistics",
    description="Returns hit/miss stats and current cache size.",
)
async def cache_stats() -> dict:
    return {"status": "ok", **summarizer_cache.stats()}
