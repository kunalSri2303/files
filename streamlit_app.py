"""
Streamlit frontend for the Text Summarizer API.
Run with: streamlit run streamlit_app.py
Requires the FastAPI backend running on http://127.0.0.1:8000
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.15);
    }

    .stat-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #667eea;
    }

    .stat-card p {
        margin: 0.3rem 0 0 0;
        color: #666;
        font-size: 0.85rem;
    }

    .summary-box {
        background: #000000;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 1.2rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    .keyword-chip {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }

    .status-ok {
        color: #00c853;
        font-weight: 600;
    }

    .status-error {
        color: #ff1744;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper ───────────────────────────────────────────────────────────────────

def check_api_health() -> dict | None:
    """Return health JSON or None if unreachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.ConnectionError:
        pass
    return None


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">📝 Text Summarizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Powered by Google Gemini — paste your text and get an AI-generated summary</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    summary_length = st.selectbox(
        "Summary Length",
        options=["short", "medium", "long"],
        index=1,
        help="short ≈ 2-3 sentences · medium ≈ 4-6 · long ≈ 8-10",
    )

    extract_keywords = st.toggle("Extract Keywords", value=True)

    st.markdown("---")

    # API status
    st.markdown("## 🔌 API Status")
    health = check_api_health()
    if health:
        st.markdown(f'Status: <span class="status-ok">● Online</span>', unsafe_allow_html=True)
        st.caption(f"Version: {health.get('version', '?')}  ·  Provider: {health.get('llm_provider', '?')}")
        st.caption(f"Cache: {'✅ Enabled' if health.get('cache_enabled') else '❌ Disabled'}")
    else:
        st.markdown(f'Status: <span class="status-error">● Offline</span>', unsafe_allow_html=True)
        st.error("Start the API first:\n```\nuvicorn main:app --reload --port 8000\n```")

    st.markdown("---")

    # Cache stats
    if health:
        st.markdown("## 📊 Cache Stats")
        try:
            cache_data = requests.get(f"{API_BASE}/cache/stats", timeout=3).json()
            c1, c2 = st.columns(2)
            c1.metric("Hits", cache_data.get("hits", 0))
            c2.metric("Misses", cache_data.get("misses", 0))
            st.caption(
                f"Size: {cache_data.get('size', 0)}/{cache_data.get('max_size', 0)}  ·  "
                f"Hit rate: {cache_data.get('hit_rate', 0):.1%}"
            )
        except Exception:
            st.caption("Could not load cache stats")


# ── Main area ────────────────────────────────────────────────────────────────

text_input = st.text_area(
    "Paste your text below",
    height=250,
    placeholder="Enter at least 50 characters of text to summarize…",
    help="Min 50 characters · Max 50,000 characters",
)

char_count = len(text_input)
st.caption(f"📏 {char_count:,} / 50,000 characters")

# ── Summarize button ─────────────────────────────────────────────────────────

if st.button("✨ Summarize", type="primary", use_container_width=True):
    if not text_input or char_count < 50:
        st.warning("⚠️ Please enter at least **50 characters** of text.")
    elif not health:
        st.error("❌ API is not running. Start it first with `uvicorn main:app --reload --port 8000`")
    else:
        with st.spinner("🤖 Gemini is summarizing your text…"):
            try:
                payload = {
                    "text": text_input,
                    "summary_length": summary_length,
                    "extract_keywords": extract_keywords,
                }
                response = requests.post(
                    f"{API_BASE}/summarize",
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()

                    # Stats row
                    st.markdown("### 📊 Summary Stats")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(
                            f'<div class="stat-card"><h3>{data["char_count_original"]:,}</h3>'
                            f'<p>Original chars</p></div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f'<div class="stat-card"><h3>{data["char_count_summary"]:,}</h3>'
                            f'<p>Summary chars</p></div>',
                            unsafe_allow_html=True,
                        )
                    with col3:
                        ratio = (
                            round(data["char_count_summary"] / data["char_count_original"] * 100, 1)
                            if data["char_count_original"] > 0
                            else 0
                        )
                        st.markdown(
                            f'<div class="stat-card"><h3>{ratio}%</h3>'
                            f'<p>Compression</p></div>',
                            unsafe_allow_html=True,
                        )
                    with col4:
                        cached_label = "✅ Yes" if data.get("cached") else "❌ No"
                        st.markdown(
                            f'<div class="stat-card"><h3>{cached_label}</h3>'
                            f'<p>Cached</p></div>',
                            unsafe_allow_html=True,
                        )

                    # Summary
                    st.markdown("### 📝 Summary")
                    st.markdown(
                        f'<div class="summary-box">{data["summary"]}</div>',
                        unsafe_allow_html=True,
                    )

                    # Keywords
                    if data.get("keywords"):
                        st.markdown("### 🏷️ Keywords")
                        chips = "".join(
                            f'<span class="keyword-chip">{kw}</span>' for kw in data["keywords"]
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                    # Raw JSON
                    with st.expander("🔍 Raw API Response"):
                        st.json(data)

                elif response.status_code == 422:
                    err = response.json()
                    st.error(f"**Validation Error:** {err.get('detail', err.get('error', 'Unknown'))}")

                elif response.status_code == 429:
                    err = response.json()
                    detail = err.get("detail", {})
                    msg = detail.get("detail", "Rate limit exceeded") if isinstance(detail, dict) else str(detail)
                    st.warning(f"⏳ **Rate Limit:** {msg}")

                elif response.status_code in (502, 503):
                    err = response.json()
                    detail = err.get("detail", {})
                    msg = detail.get("detail", "LLM service error") if isinstance(detail, dict) else str(detail)
                    st.error(f"🔌 **Service Error:** {msg}")

                else:
                    st.error(f"❌ Unexpected error (HTTP {response.status_code})")
                    st.json(response.json())

            except requests.ConnectionError:
                st.error("❌ Cannot connect to the API. Make sure it's running on port 8000.")
            except requests.Timeout:
                st.error("⏱️ Request timed out. The LLM might be overloaded — try again.")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#aaa; font-size:0.85rem;">'
    'Text Summarizer API · Built with FastAPI + Gemini + Streamlit'
    '</p>',
    unsafe_allow_html=True,
)
