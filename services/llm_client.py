import asyncio
from typing import Optional
import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    DeadlineExceeded,
    InvalidArgument,
    PermissionDenied,
    GoogleAPIError,
)
from utils.config import get_settings
from utils.logger import llm_logger

settings = get_settings()

# Prompt templates keyed by summary length
SUMMARY_PROMPTS: dict[str, str] = {
    "short": (
        "Summarize the following text in 2-3 sentences. "
        "Capture only the single most important idea. Be extremely concise."
    ),
    "medium": (
        "Summarize the following text in a medium-length format (4-6 sentences). "
        "Cover the main points clearly and concisely, preserving key context."
    ),
    "long": (
        "Summarize the following text in a comprehensive format (8-10 sentences). "
        "Cover all significant points, supporting details, and conclusions. "
        "Maintain the logical flow of the original."
    ),
}

KEYWORD_PROMPT = (
    "Additionally, after the summary, on a new line write exactly: "
    "KEYWORDS: followed by a comma-separated list of 5-8 key topics or terms from the text."
)


class LLMClient:
    """
    Async wrapper around the Google Gemini generative AI client.
    Handles prompt construction, retries with exponential backoff,
    and maps Gemini-specific exceptions to clean Python errors.
    """

    def __init__(self):
        self._model: Optional[genai.GenerativeModel] = None

    def _get_model(self) -> genai.GenerativeModel:
        if self._model is None:
            if not settings.GEMINI_API_KEY:
                raise PermissionDenied(
                    "GEMINI_API_KEY is not set. Add it to your .env file."
                )
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._model = genai.GenerativeModel(
                model_name=settings.LLM_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.LLM_TEMPERATURE,
                    max_output_tokens=settings.LLM_MAX_TOKENS,
                ),
                system_instruction=(
                    "You are an expert text summarizer. Your task is to produce accurate, "
                    "neutral, and well-structured summaries. Never add information not present "
                    "in the original text. Use clear, professional English."
                ),
            )
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_content: str,
        max_retries: int = 2,
    ) -> str:
        """
        Send a prompt to Gemini and return the response text.
        Retries on rate-limit and timeout errors with exponential backoff.

        Note: Gemini merges system + user prompts into a single `contents` call.
        The system_instruction is set at model level; here we combine the
        length-specific instruction with the user content in one user turn.
        """
        model = self._get_model()
        full_prompt = f"{system_prompt}\n\n{user_content}"
        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(max_retries + 1):
            try:
                llm_logger.info(
                    "Gemini request started",
                    extra={
                        "model": settings.LLM_MODEL,
                        "attempt": attempt + 1,
                        "input_chars": len(full_prompt),
                    },
                )

                response = await model.generate_content_async(full_prompt)

                content = response.text or ""
                llm_logger.info(
                    "Gemini request succeeded",
                    extra={
                        "model": settings.LLM_MODEL,
                        "output_chars": len(content),
                    },
                )
                return content.strip()

            except ResourceExhausted as e:
                # 429 — quota / rate limit
                last_error = e
                wait = 2 ** attempt
                llm_logger.warning(
                    "Gemini rate limited, backing off",
                    extra={"attempt": attempt + 1, "wait_seconds": wait},
                )
                await asyncio.sleep(wait)

            except DeadlineExceeded as e:
                # 504 — request timeout
                last_error = e
                llm_logger.error("Gemini request timed out", extra={"attempt": attempt + 1})
                if attempt < max_retries:
                    await asyncio.sleep(1)

            except PermissionDenied as e:
                # 403 — bad API key, do not retry
                llm_logger.error("Gemini authentication failed", extra={"error": str(e)})
                raise

            except InvalidArgument as e:
                # 400 — bad request, do not retry
                llm_logger.error("Gemini invalid argument", extra={"error": str(e)})
                raise

            except GoogleAPIError as e:
                last_error = e
                llm_logger.error(
                    "Gemini API error",
                    extra={"attempt": attempt + 1, "error": str(e)},
                )
                if attempt < max_retries:
                    await asyncio.sleep(1)

        raise last_error

    async def close(self):
        """No persistent connection to close for Gemini; kept for API compatibility."""
        self._model = None


# Singleton instance
llm_client = LLMClient()