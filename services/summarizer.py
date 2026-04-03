from typing import Optional
from schemas.summarize import SummarizeRequest, SummarizeResponse, SummaryLength
from services.llm_client import llm_client, SUMMARY_PROMPTS, KEYWORD_PROMPT
from utils.cache import summarizer_cache
from utils.config import get_settings
from utils.logger import app_logger

settings = get_settings()

SYSTEM_BASE = (
    "You are an expert text summarizer. Your task is to produce accurate, "
    "neutral, and well-structured summaries. Never add information not present "
    "in the original text. Use clear, professional English."
)


class SummarizationService:
    """
    Orchestrates the summarization pipeline:
      1. Check cache
      2. Build prompt
      3. Call LLM
      4. Parse response (summary + optional keywords)
      5. Cache result
    """

    async def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        cache_key = summarizer_cache.make_key(request.text, request.summary_length.value)

        # ── Cache check ──────────────────────────────────────────────
        if settings.CACHE_ENABLED:
            cached = summarizer_cache.get(cache_key)
            if cached is not None:
                app_logger.info(
                    "Returning cached summary",
                    extra={"length": request.summary_length, "cache_key": cache_key[:16]},
                )
                cached["cached"] = True
                return SummarizeResponse(**cached)

        # ── Build prompts ─────────────────────────────────────────────
        length_instruction = SUMMARY_PROMPTS[request.summary_length.value]
        system_prompt = f"{SYSTEM_BASE}\n\n{length_instruction}"

        if request.extract_keywords:
            system_prompt += f"\n\n{KEYWORD_PROMPT}"

        user_content = f"TEXT TO SUMMARIZE:\n\n{request.text}"

        # ── LLM call ──────────────────────────────────────────────────
        app_logger.info(
            "Summarization request",
            extra={
                "length": request.summary_length,
                "input_chars": len(request.text),
                "extract_keywords": request.extract_keywords,
            },
        )

        raw_response = await llm_client.complete(
            system_prompt=system_prompt,
            user_content=user_content,
        )

        # ── Parse response ────────────────────────────────────────────
        summary, keywords = self._parse_response(raw_response, request.extract_keywords)

        result = {
            "summary": summary,
            "length": request.summary_length,
            "status": "success",
            "char_count_original": len(request.text),
            "char_count_summary": len(summary),
            "keywords": keywords,
            "cached": False,
        }

        # ── Cache result ──────────────────────────────────────────────
        if settings.CACHE_ENABLED:
            summarizer_cache.set(cache_key, result)

        return SummarizeResponse(**result)

    @staticmethod
    def _parse_response(
        raw: str, expect_keywords: bool
    ) -> tuple[str, Optional[list[str]]]:
        """
        Split raw LLM response into (summary_text, keywords_list).
        Keywords section is identified by the 'KEYWORDS:' sentinel.
        """
        keywords: Optional[list[str]] = None

        if expect_keywords and "KEYWORDS:" in raw:
            parts = raw.split("KEYWORDS:", 1)
            summary = parts[0].strip()
            kw_raw = parts[1].strip()
            keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
        else:
            summary = raw.strip()

        return summary, keywords


# Singleton
summarization_service = SummarizationService()
