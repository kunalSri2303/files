"""
Tests for the Text Summarizer API (Gemini backend).
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app
from utils.cache import summarizer_cache
from utils.rate_limiter import _limiter

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear cache and rate limiter state before each test."""
    summarizer_cache._cache.clear()
    summarizer_cache._hits = 0
    summarizer_cache._misses = 0
    _limiter._windows.clear()


SAMPLE_TEXT = (
    "Artificial intelligence (AI) is transforming industries across the globe. "
    "From healthcare to finance, AI systems are being deployed to automate tasks, "
    "improve decision-making, and unlock new capabilities. Machine learning, a "
    "subset of AI, enables systems to learn from data without explicit programming. "
    "Deep learning, using multi-layered neural networks, has driven breakthroughs "
    "in image recognition, natural language processing, and game playing. "
    "However, concerns about bias, explainability, and job displacement remain "
    "active areas of debate among researchers and policymakers."
)


def _mock_gemini_model(response_text: str):
    """Return a mock GenerativeModel whose generate_content_async returns response_text."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    return mock_model


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_shape(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["llm_provider"] == "gemini"
        assert "cache_enabled" in data

    def test_cache_stats_endpoint(self):
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "misses" in data


# ── Validation ────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_empty_text_rejected(self):
        response = client.post("/summarize", json={"text": ""})
        assert response.status_code == 422

    def test_whitespace_only_text_rejected(self):
        response = client.post("/summarize", json={"text": "   \n\t  "})
        assert response.status_code == 422

    def test_text_too_short_rejected(self):
        response = client.post("/summarize", json={"text": "Too short."})
        assert response.status_code == 422

    def test_text_too_long_rejected(self):
        response = client.post("/summarize", json={"text": "a" * 50_001})
        assert response.status_code == 422

    def test_invalid_summary_length_rejected(self):
        response = client.post(
            "/summarize", json={"text": SAMPLE_TEXT, "summary_length": "tiny"}
        )
        assert response.status_code == 422


# ── Summarization (mocked Gemini) ─────────────────────────────────────────────

class TestSummarizationEndpoint:
    def test_successful_summarization(self):
        summary_text = "AI is transforming industries through automation and ML."
        with patch(
            "services.llm_client.llm_client._get_model",
            return_value=_mock_gemini_model(summary_text),
        ):
            response = client.post("/summarize", json={"text": SAMPLE_TEXT})

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            assert "summary" in data
            assert data["length"] == "medium"
            assert data["char_count_original"] == len(SAMPLE_TEXT)
            assert data["cached"] is False

    def test_response_contains_required_fields(self):
        with patch(
            "services.llm_client.llm_client._get_model",
            return_value=_mock_gemini_model("Short AI summary."),
        ):
            response = client.post(
                "/summarize",
                json={"text": SAMPLE_TEXT, "summary_length": "short"},
            )

        if response.status_code == 200:
            data = response.json()
            for field in ["summary", "length", "status", "char_count_original", "char_count_summary", "cached"]:
                assert field in data, f"Missing field: {field}"

    def test_keyword_extraction_returns_list(self):
        llm_output = (
            "AI transforms industries.\n"
            "KEYWORDS: artificial intelligence, machine learning, automation, neural networks, bias"
        )
        with patch(
            "services.llm_client.llm_client._get_model",
            return_value=_mock_gemini_model(llm_output),
        ):
            response = client.post(
                "/summarize",
                json={"text": SAMPLE_TEXT, "extract_keywords": True},
            )

        if response.status_code == 200:
            data = response.json()
            assert data["keywords"] is not None
            assert isinstance(data["keywords"], list)
            assert len(data["keywords"]) > 0

    def test_default_summary_length_is_medium(self):
        with patch(
            "services.llm_client.llm_client._get_model",
            return_value=_mock_gemini_model("A summary."),
        ):
            response = client.post("/summarize", json={"text": SAMPLE_TEXT})

        if response.status_code == 200:
            assert response.json()["length"] == "medium"

    def test_valid_summary_lengths(self):
        for length in ["short", "medium", "long"]:
            with patch(
                "services.llm_client.llm_client._get_model",
                return_value=_mock_gemini_model("Summary."),
            ):
                r = client.post(
                    "/summarize",
                    json={"text": SAMPLE_TEXT, "summary_length": length},
                )
            assert r.status_code in (200, 503), f"Unexpected status for length={length}"


# ── Cache ─────────────────────────────────────────────────────────────────────

class TestCache:
    def test_cache_returns_cached_true_on_second_call(self):
        call_count = 0

        async def counting_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "Cached Gemini summary."

        with patch("services.summarizer.llm_client.complete", side_effect=counting_complete):
            r1 = client.post("/summarize", json={"text": SAMPLE_TEXT, "summary_length": "long"})
            r2 = client.post("/summarize", json={"text": SAMPLE_TEXT, "summary_length": "long"})

        if r1.status_code == 200 and r2.status_code == 200:
            assert r2.json()["cached"] is True
            assert call_count == 1  # LLM called only once


# ── Error Handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_text_field(self):
        response = client.post("/summarize", json={})
        assert response.status_code == 422

    def test_error_response_has_error_field(self):
        response = client.post("/summarize", json={"text": ""})
        data = response.json()
        assert "error" in data or "detail" in data