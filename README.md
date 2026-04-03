# Text Summarizer API

A production-ready REST API that uses an LLM (OpenAI) to summarize long text.  
Built with **FastAPI**, **Pydantic v2**, async/await throughout, in-memory LRU caching, sliding-window rate limiting, and structured JSON logging.

---

## Folder Structure

```
text-summarizer-api/
├── main.py                  # App factory, middleware, exception handlers
├── requirements.txt
├── .env.example
│
├── routers/
│   ├── __init__.py
│   ├── health.py            # GET /health, GET /cache/stats
│   └── summarize.py         # POST /summarize
│
├── services/
│   ├── __init__.py
│   ├── llm_client.py        # Async OpenAI wrapper with retry logic
│   └── summarizer.py        # Orchestration: cache → LLM → parse → cache
│
├── schemas/
│   ├── __init__.py
│   └── summarize.py         # Pydantic request/response models
│
├── utils/
│   ├── __init__.py
│   ├── config.py            # Pydantic-settings configuration
│   ├── logger.py            # JSON structured logging
│   ├── cache.py             # TTL + LRU in-memory cache
│   └── rate_limiter.py      # Sliding-window per-IP rate limiter
│
└── tests/
    ├── __init__.py
    └── test_api.py          # Unit + integration tests (mocked LLM)
```

---

## Quick Start

### 1. Clone & create a virtual environment

```bash
git clone <repo-url>
cd text-summarizer-api
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 4. Run the server

```bash
# Development (auto-reload)
DEBUG=true python main.py

# OR with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API is now available at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "llm_provider": "openai",
  "cache_enabled": true
}
```

---

### `POST /summarize`

**Request body**

| Field             | Type    | Required | Default  | Description                             |
|-------------------|---------|----------|----------|-----------------------------------------|
| `text`            | string  | ✅       | —        | Text to summarize (50–50,000 characters)|
| `summary_length`  | enum    | ❌       | `medium` | `short` / `medium` / `long`             |
| `extract_keywords`| boolean | ❌       | `false`  | Whether to extract key topics           |

**curl – medium summary**

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence (AI) is transforming industries across the globe. From healthcare to finance, AI systems are being deployed to automate tasks, improve decision-making, and unlock new capabilities. Machine learning, a subset of AI, enables systems to learn from data without explicit programming. Deep learning, using multi-layered neural networks, has driven breakthroughs in image recognition, natural language processing, and game playing. However, concerns about bias, explainability, and job displacement remain active areas of debate among researchers and policymakers.",
    "summary_length": "medium"
  }'
```

**Response**

```json
{
  "summary": "AI is revolutionizing industries such as healthcare and finance by automating processes and enhancing decision-making. Machine learning allows systems to learn from data, while deep learning has achieved major breakthroughs in vision and language tasks. Despite these advances, ethical concerns around bias, transparency, and workforce impact continue to be widely discussed.",
  "length": "medium",
  "status": "success",
  "char_count_original": 643,
  "char_count_summary": 318,
  "keywords": null,
  "cached": false
}
```

---

**curl – short summary with keywords**

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "...",
    "summary_length": "short",
    "extract_keywords": true
  }'
```

```json
{
  "summary": "AI is reshaping industries globally, with machine learning and deep learning driving major advances—though ethical concerns persist.",
  "length": "short",
  "status": "success",
  "char_count_original": 643,
  "char_count_summary": 128,
  "keywords": ["artificial intelligence", "machine learning", "deep learning", "neural networks", "bias"],
  "cached": false
}
```

---

### `GET /cache/stats`

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "status": "ok",
  "size": 4,
  "max_size": 500,
  "ttl_seconds": 3600,
  "hits": 7,
  "misses": 4,
  "hit_rate": 0.636
}
```

---

## Error Responses

All errors follow this shape:

```json
{
  "status": "error",
  "error": "Human-readable error type",
  "detail": "More specific message",
  "code": 422
}
```

| HTTP Code | Cause                                   |
|-----------|-----------------------------------------|
| `422`     | Validation failed (empty/too long text) |
| `429`     | Rate limit exceeded (10 req/60 s/IP)    |
| `502`     | OpenAI API error                        |
| `503`     | Invalid API key or upstream rate limit  |
| `504`     | LLM request timeout                     |
| `500`     | Unexpected server error                 |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked LLM calls — no real API key required.

---

## Configuration Reference

All settings can be overridden via `.env` or environment variables:

| Variable               | Default         | Description                       |
|------------------------|-----------------|-----------------------------------|
| `OPENAI_API_KEY`       | *(required)*    | Your OpenAI API key               |
| `LLM_MODEL`            | `gpt-4o-mini`   | Model to use                      |
| `LLM_MAX_TOKENS`       | `1024`          | Max tokens in LLM response        |
| `LLM_TEMPERATURE`      | `0.3`           | Sampling temperature              |
| `RATE_LIMIT_REQUESTS`  | `10`            | Requests per window per IP        |
| `RATE_LIMIT_WINDOW`    | `60`            | Window size in seconds            |
| `CACHE_ENABLED`        | `true`          | Toggle in-memory caching          |
| `CACHE_TTL`            | `3600`          | Cache entry lifetime in seconds   |
| `CACHE_MAX_SIZE`       | `500`           | Max cached entries (LRU eviction) |
| `DEBUG`                | `false`         | Enable uvicorn auto-reload        |
