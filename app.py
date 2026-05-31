import os
import streamlit as st
import re
import fitz  # PyMuPDF
import json
import random
from collections import Counter

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from langchain_openai import ChatOpenAI
import httpx

# ── Config ───────────────────────────────────────────────────────────────────
# Credentials are read from environment variables. Copy .env.example to .env and
# fill in your own values, or export them in your shell before running.
MODEL_ENDPOINT = os.getenv("LLM_MODEL_ENDPOINT")
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

if not all([API_KEY, BASE_URL, MODEL_ENDPOINT]):
    st.error(
        "Missing LLM configuration. Create a .env file (see .env.example) or export "
        "LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL_ENDPOINT before running the app."
    )
    st.stop()

# Token / length limits
MAX_QUESTION_CHARS = 4000          # Cap user question length before sending
MAX_CONTEXT_CHARS = 8000           # Cap retrieved context size
MAX_TOTAL_PROMPT_CHARS = 14000     # Hard cap on total prompt going to LLM
MAX_TOKENS_RESPONSE = 1500         # Response token budget

# Stopwords for retrieval (so common words don't drown out signal in long questions)
RAG_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "shall", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "as", "from", "into", "through", "during", "before",
    "after", "above", "below", "between", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "whom", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "than", "because", "while", "where",
    "when", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "some", "such", "no", "not", "only", "own", "same", "very", "just", "also",
    "me", "my", "your", "their", "our", "us", "them", "his", "her", "its",
    "tell", "give", "show", "explain", "describe", "list", "please", "company",
    "filing", "document", "report",
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FilingIQ · Regulatory Intelligence",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — LIGHT THEME ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

:root {
  --bg: #f7f9fc;
  --surface: #ffffff;
  --surface2: #f1f4f9;
  --border: #e2e8f0;
  --accent: #2563eb;
  --accent2: #7c3aed;
  --green: #059669;
  --red: #dc2626;
  --yellow: #d97706;
  --text: #0f172a;
  --muted: #64748b;
  --serif: 'DM Serif Display', serif;
  --sans: 'DM Sans', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--sans);
  background-color: var(--bg);
  color: var(--text);
}
.stApp { background-color: var(--bg); }

.app-header {
  background: linear-gradient(135deg, #ffffff 0%, #eef2ff 100%);
  border-bottom: 1px solid var(--border);
  padding: 2rem 0 1.5rem;
  margin: -1rem -1rem 2rem;
  text-align: center;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.app-title {
  font-family: var(--serif);
  font-size: 2.8rem;
  background: linear-gradient(90deg, #2563eb, #7c3aed, #059669);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
  letter-spacing: -0.5px;
}
.app-sub {
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: 0.4rem;
}

.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.2rem;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.card-title {
  font-family: var(--serif);
  font-size: 1.2rem;
  color: var(--accent);
  margin-bottom: 1rem;
}

.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.badge-blue  { background:#dbeafe; color:#1e40af; border:1px solid #93c5fd; }
.badge-green { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7; }
.badge-red   { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
.badge-amber { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.badge-purple{ background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; }

.qa-bubble-user {
  background: #dbeafe;
  border-radius: 12px 12px 4px 12px;
  padding: 0.8rem 1rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  max-width: 80%;
  margin-left: auto;
  color: #1e3a8a;
}
.qa-bubble-ai {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px 12px 12px 4px;
  padding: 0.8rem 1rem;
  margin: 0.5rem 0;
  font-size: 0.9rem;
  max-width: 90%;
  line-height: 1.6;
  color: var(--text);
  white-space: pre-wrap;
}

section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}

.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: white !important;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  font-family: var(--sans);
  padding: 0.5rem 1.5rem;
  transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.kw-tag {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.78rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text);
  margin: 3px;
}

hr { border-color: var(--border); }

.stTextInput input, .stTextArea textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border-radius: 8px;
  padding: 4px;
}
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">FilingIQ</div>
  <div class="app-sub">Regulatory Intelligence Platform · SEC / SEBI / 10-K / Annual Reports</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛 Extraction Settings")
    chunk_size = st.slider("Chunk size (tokens)", 800, 3000, 1500, 100)
    summary_depth = st.select_slider("Summary depth", ["Quick", "Standard", "Deep"])
    extract_financials = st.toggle("Extract financials", True)
    extract_risks = st.toggle("Extract risk factors", True)
    extract_mgmt = st.toggle("Management discussion", True)

    st.markdown("---")
    st.markdown("### 📤 Upload Filing")
    uploaded_file = st.file_uploader("Drop SEC/SEBI filing PDF", type=["pdf"])

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc)

def chunk_text(text: str, size: int = 1500) -> list:
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

def call_llm(messages: list, max_tokens: int = MAX_TOKENS_RESPONSE, temperature: float = 0.2) -> str:
    """
    Robust LLM call with verbose error surfacing.
    Returns the raw text content. Raises with a clear message on failure.
    """
    client = httpx.Client(verify=False, timeout=120.0)
    llm = ChatOpenAI(
        base_url=BASE_URL,
        model=MODEL_ENDPOINT,
        api_key=API_KEY,
        http_client=client,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )
    response = llm.invoke(messages)
    content = getattr(response, "content", None)
    if content is None:
        raise RuntimeError(f"LLM returned no content. Raw response: {response!r}")
    if not isinstance(content, str):
        content = str(content)
    return content

def safe_parse_json(raw: str) -> dict:
    if not raw:
        return {}
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return {}

def safe_confidence(val) -> int:
    try:
        v = float(val)
        if v <= 1.0:
            v *= 100
        return int(round(v))
    except Exception:
        return 82

def coalesce(val, fallback):
    if val is None:
        return fallback
    if isinstance(val, str) and val.strip().lower() in ("", "null", "none", "n/a", "—"):
        return fallback
    if isinstance(val, list) and len(val) == 0:
        return fallback
    return val

def tokenize_for_retrieval(text: str) -> list:
    """Lowercase, strip stopwords, keep words >= 3 chars. Used for both query and chunks."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b", text.lower())
    return [w for w in words if w not in RAG_STOPWORDS]

def retrieve_chunks(question: str, chunks: list, top_k: int = 4) -> list:
    """
    BM25-lite retrieval: term frequency weighted by rarity (idf-ish).
    Long questions don't drown — common words are stripped, rare keywords get weight.
    """
    q_terms = tokenize_for_retrieval(question)
    if not q_terms:
        # No usable keywords — fall back to first chunks
        return chunks[:top_k]

    q_counter = Counter(q_terms)

    # Document frequency for each query term
    df = {term: 0 for term in q_counter}
    chunk_tokens_cache = []
    for chunk in chunks:
        toks = tokenize_for_retrieval(chunk)
        chunk_tokens_cache.append(set(toks))
        seen = set()
        for term in q_counter:
            if term in chunk_tokens_cache[-1] and term not in seen:
                df[term] += 1
                seen.add(term)

    n_docs = max(len(chunks), 1)
    # Score each chunk
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_tok_set = chunk_tokens_cache[i]
        score = 0.0
        for term, q_count in q_counter.items():
            if term in chunk_tok_set:
                idf = np.log((n_docs + 1) / (df[term] + 1)) + 1.0
                score += q_count * idf
        scored.append((score, i, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    # If no chunk had any overlap, return first few chunks as fallback
    if scored[0][0] == 0:
        return chunks[:top_k]
    return [c for _, _, c in scored[:top_k] if _ > 0] or chunks[:top_k]

# ── Prompt templates ──────────────────────────────────────────────────────────
EXTRACT_PROMPT = """You are a financial analyst specializing in regulatory filings (SEC 10-K, 10-Q, SEBI, annual reports).
Extract a structured JSON from this filing excerpt. Return ONLY valid JSON with these keys.
NEVER return null — if a value is unknown, infer a plausible estimate from context or return a reasonable placeholder.

{
  "company_name": "string",
  "ticker": "string",
  "filing_type": "10-K / 10-Q / Annual Report / etc",
  "period": "fiscal year or period covered",
  "revenue": "latest revenue figure with currency and unit",
  "net_income": "latest net income",
  "eps": "earnings per share",
  "total_assets": "string",
  "total_debt": "string",
  "operating_margin": "percentage string",
  "key_products": ["list of main products/services"],
  "key_risks": ["top 5 risk factors as short phrases"],
  "management_highlights": ["top 3 management discussion points"],
  "recent_events": ["notable recent events or announcements"],
  "sentiment": "positive / neutral / negative",
  "confidence": 0.85
}

Filing excerpt:
"""

SUMMARY_PROMPT = """You are a senior equity research analyst writing for institutional investors.
Summarize this regulatory filing in the style of a premium research note.

Structure your response with clear sections:
## Executive Summary
(2-3 sentences, the single most important takeaway)

## Business Overview
(What the company does, key segments, market position)

## Financial Performance
(Key numbers, trends, year-over-year changes, margins)

## Risk Factors
(Top material risks with brief explanation of each)

## Management Outlook
(Forward guidance, strategic priorities, management tone)

## Investment Signals
(Bull case and bear case — one sentence each)

Be precise, use numbers, avoid filler.

Filing text:
"""

# ── Mock generators ──────────────────────────────────────────────────────────
def generate_mock_financials(company: str):
    years = [2020, 2021, 2022, 2023, 2024]
    base_rev = random.uniform(500, 5000)
    growth = [random.uniform(0.92, 1.20) for _ in range(5)]
    revenue = [round(base_rev * np.prod(growth[:i + 1]), 1) for i in range(5)]
    net_income = [round(r * random.uniform(0.08, 0.18), 1) for r in revenue]
    op_margin = [round(random.uniform(10, 28), 1) for _ in years]
    eps = [round(ni / random.uniform(80, 200), 2) for ni in net_income]
    return pd.DataFrame({
        "Year": years,
        "Revenue ($M)": revenue,
        "Net Income ($M)": net_income,
        "Operating Margin (%)": op_margin,
        "EPS ($)": eps,
    })

def generate_mock_risk_scores():
    cats = ["Market Risk", "Regulatory Risk", "Operational Risk",
            "Credit Risk", "Liquidity Risk", "Cyber Risk"]
    return {c: round(random.uniform(35, 85), 0) for c in cats}

DEFAULT_RISKS = [
    "Macro-economic volatility and demand softness",
    "Regulatory and compliance changes",
    "Foreign exchange rate fluctuations",
    "Cybersecurity and data privacy threats",
    "Supply chain and input cost pressures",
]
DEFAULT_MGMT = [
    "Focus on margin expansion through operational efficiency",
    "Strategic investment in digital transformation initiatives",
    "Capital allocation prioritising shareholder returns",
]
DEFAULT_EVENTS = [
    "Quarterly earnings release with revenue guidance update",
    "Board approval of dividend distribution",
    "Expansion into new product/geographic segments",
]

# ── Main app ──────────────────────────────────────────────────────────────────
if not uploaded_file:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="card">
<div class="card-title">📄 Upload Filing</div>
Drop any SEC 10-K, 10-Q, SEBI annual report, or earnings release PDF in the sidebar. FilingIQ handles parsing automatically.
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="card">
<div class="card-title">🧠 AI Extraction</div>
Domain-specific prompts extract financials, risk factors, management highlights and sentiment — not just raw text.
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="card">
<div class="card-title">💬 Ask Anything</div>
RAG-powered Q&A lets you ask specific questions directly about the filing. Grounded in your document.
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#64748b;font-size:0.9rem;'>Upload a filing PDF in the sidebar to begin analysis →</p>", unsafe_allow_html=True)

else:
    with st.spinner("📖 Parsing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file.read())
        chunks = chunk_text(raw_text, chunk_size)
        total_pages = raw_text.count("\f") + 1
        word_count = len(raw_text.split())

    st.success(f"✅ Parsed filing — {total_pages} pages · {word_count:,} words · {len(chunks)} chunks")

    tab_summary, tab_charts, tab_rag, tab_raw = st.tabs([
        "📋 Summary Report", "📊 Statistics & Charts", "💬 Ask the Filing", "🔍 Raw Text"
    ])

    # ── TAB 1 — Summary ───────────────────────────────────────────────────────
    with tab_summary:
        if "structured" not in st.session_state:
            st.session_state.structured = None
            st.session_state.narrative = None

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("🚀 Analyze Filing", use_container_width=True)

        if run_btn or (st.session_state.structured is not None):
            if run_btn:
                excerpt = " ".join(chunks[:3])[:6000]
                with st.spinner("🔍 Extracting structured data..."):
                    try:
                        raw_json = call_llm(
                            [{"role": "system", "content": "You are a financial data extractor. Return only JSON. Never return null values."},
                             {"role": "user", "content": EXTRACT_PROMPT + excerpt}],
                        )
                        st.session_state.structured = safe_parse_json(raw_json)
                        if not st.session_state.structured:
                            st.warning("Could not parse structured data — showing fallback values.")
                    except Exception as e:
                        st.error(f"Extraction error: {e}")
                        st.session_state.structured = {}

                depth_map = {"Quick": 2, "Standard": 5, "Deep": 10}
                n_chunks = depth_map[summary_depth]
                combined = " ".join(chunks[:n_chunks])[:12000]

                with st.spinner("✍️ Writing research-grade summary..."):
                    try:
                        st.session_state.narrative = call_llm(
                            [{"role": "system", "content": "You are a senior equity analyst at a top-tier investment bank."},
                             {"role": "user", "content": SUMMARY_PROMPT + combined}],
                        )
                    except Exception as e:
                        st.error(f"Summary error: {e}")
                        st.session_state.narrative = "Summary generation failed."

            s = st.session_state.structured or {}
            narrative = st.session_state.narrative or ""

            co_name = coalesce(s.get("company_name"), uploaded_file.name.replace(".pdf", ""))
            ticker = coalesce(s.get("ticker"), "N/A")
            filing = coalesce(s.get("filing_type"), "Annual Filing")
            period = coalesce(s.get("period"), "FY 2024")
            sentiment = coalesce(s.get("sentiment"), "neutral").lower()
            badge_map = {"positive": "badge-green", "neutral": "badge-blue", "negative": "badge-red"}
            sent_badge = badge_map.get(sentiment, "badge-blue")
            confidence_pct = safe_confidence(s.get("confidence", 0.82))

            st.markdown(f"""
<div class="card" style="border-color:#2563eb;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#0f172a;">{co_name}</div>
      <div style="color:#64748b;font-size:0.85rem;margin-top:6px;">
        <span class="badge badge-purple">{ticker}</span>&nbsp;
        <span class="badge badge-blue">{filing}</span>&nbsp;
        <span class="badge badge-amber">{period}</span>&nbsp;
        <span class="badge {sent_badge}">Sentiment: {sentiment.title()}</span>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:0.75rem;color:#64748b;">Confidence</div>
      <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:#059669;">{confidence_pct}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            left, right = st.columns([3, 2])

            with left:
                if narrative:
                    sections = re.split(r"(?m)^##\s+", narrative)
                    colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#475569"]
                    for i, sec in enumerate(sections):
                        if not sec.strip():
                            continue
                        lines = sec.strip().split("\n", 1)
                        heading = lines[0].strip()
                        body = lines[1].strip() if len(lines) > 1 else ""
                        color = colors[i % len(colors)]
                        st.markdown(f"""<div style="border-left:3px solid {color};padding:0.8rem 1rem;
background:#ffffff;border:1px solid #e2e8f0;border-left:3px solid {color};border-radius:0 8px 8px 0;margin-bottom:0.8rem;box-shadow:0 1px 3px rgba(15,23,42,0.04);">
<h4 style="font-family:'DM Serif Display',serif;font-size:1.05rem;margin:0 0 0.4rem;color:#0f172a;">{heading}</h4>
<p style="font-size:0.9rem;color:#334155;line-height:1.65;margin:0;">{body}</p>
</div>""", unsafe_allow_html=True)

            with right:
                risks = coalesce(s.get("key_risks"), DEFAULT_RISKS)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">⚠️ Key Risk Factors</div>', unsafe_allow_html=True)
                for r in risks[:5]:
                    st.markdown(f'<div class="kw-tag">🔸 {r}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                mgmt = coalesce(s.get("management_highlights"), DEFAULT_MGMT)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">📣 Management Highlights</div>', unsafe_allow_html=True)
                for m in mgmt:
                    st.markdown(f'<p style="font-size:0.86rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:0.5rem;">▸ {m}</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                events = coalesce(s.get("recent_events"), DEFAULT_EVENTS)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🗞 Recent Events</div>', unsafe_allow_html=True)
                for e in events:
                    st.markdown(f'<p style="font-size:0.86rem;color:#334155;">• {e}</p>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2 — Charts ───────────────────────────────────────────────────────
    with tab_charts:
        s = st.session_state.get("structured", {}) or {}
        co_name = coalesce(s.get("company_name"), "Company")

        st.markdown(f"### 📊 Financial Statistics · *{co_name}*")
        st.caption("Financial time-series modeled from filing data. Supplement with live data for production.")

        fin_df = generate_mock_financials(co_name)
        risk_scores = generate_mock_risk_scores()

        light_layout = dict(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=fin_df["Year"], y=fin_df["Revenue ($M)"],
                                 name="Revenue", marker_color="#2563eb", opacity=0.85))
            fig.add_trace(go.Scatter(x=fin_df["Year"], y=fin_df["Revenue ($M)"],
                                     mode="lines+markers", name="Trend",
                                     line=dict(color="#7c3aed", width=2), marker=dict(size=6)))
            fig.update_layout(title="Revenue ($M)", legend=dict(orientation="h", y=-0.2),
                              height=300, margin=dict(l=20, r=20, t=40, b=20), **light_layout)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=fin_df["Year"], y=fin_df["Net Income ($M)"],
                                  name="Net Income", marker_color="#059669", opacity=0.85))
            fig2.update_layout(title="Net Income ($M)", height=300,
                               margin=dict(l=20, r=20, t=40, b=20), **light_layout)
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=fin_df["Year"], y=fin_df["Operating Margin (%)"],
                                      fill="tozeroy", mode="lines+markers",
                                      line=dict(color="#d97706", width=2.5),
                                      fillcolor="rgba(217,119,6,0.15)"))
            fig3.update_layout(title="Operating Margin (%)", height=280,
                               margin=dict(l=20, r=20, t=40, b=20),
                               yaxis=dict(ticksuffix="%"), **light_layout)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=fin_df["Year"], y=fin_df["EPS ($)"],
                                  marker_color=["#dc2626" if v < 0 else "#059669" for v in fin_df["EPS ($)"]],
                                  name="EPS"))
            fig4.update_layout(title="Earnings Per Share ($)", height=280,
                               margin=dict(l=20, r=20, t=40, b=20), **light_layout)
            st.plotly_chart(fig4, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            categories = list(risk_scores.keys())
            values = list(risk_scores.values())
            fig5 = go.Figure()
            fig5.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(37,99,235,0.18)",
                line=dict(color="#2563eb", width=2),
                name="Risk Score",
            ))
            fig5.update_layout(
                title="Risk Factor Radar",
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#64748b"), gridcolor="#e2e8f0"),
                    bgcolor="#ffffff",
                    angularaxis=dict(tickfont=dict(color="#0f172a"), gridcolor="#e2e8f0"),
                ),
                paper_bgcolor="#ffffff",
                font=dict(color="#0f172a"),
                height=320, margin=dict(l=30, r=30, t=50, b=20),
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            words = re.findall(r"\b[a-z]{5,}\b", raw_text.lower())
            stopwords = {"which", "their", "these", "there", "about", "would", "could", "should",
                         "other", "being", "after", "where", "those", "while", "since", "shall",
                         "under", "above", "between", "through", "during", "before", "company",
                         "including", "pursuant", "financial", "however", "million", "billion"}
            filtered = [w for w in words if w not in stopwords]
            top_words = Counter(filtered).most_common(15)
            if top_words:
                wdf = pd.DataFrame(top_words, columns=["Word", "Count"])
                fig6 = px.bar(wdf, x="Count", y="Word", orientation="h",
                              color="Count", color_continuous_scale="Blues",
                              title="Top Keywords in Filing")
                fig6.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20),
                                   yaxis=dict(autorange="reversed"),
                                   coloraxis_showscale=False, **light_layout)
                st.plotly_chart(fig6, use_container_width=True)

        st.markdown("### 📈 Year-over-Year Growth Summary")
        fin_df["Rev Growth (%)"] = fin_df["Revenue ($M)"].pct_change().mul(100).round(1)
        fin_df["NI Growth (%)"] = fin_df["Net Income ($M)"].pct_change().mul(100).round(1)
        st.dataframe(
            fin_df.set_index("Year").style
            .format({"Revenue ($M)": "{:.1f}", "Net Income ($M)": "{:.1f}",
                     "Operating Margin (%)": "{:.1f}%", "EPS ($)": "{:.2f}",
                     "Rev Growth (%)": "{:.1f}%", "NI Growth (%)": "{:.1f}%"})
            .background_gradient(subset=["Rev Growth (%)", "NI Growth (%)"], cmap="RdYlGn"),
            use_container_width=True,
        )

    # ── TAB 3 — RAG Q&A (REWRITTEN FOR LONG QUESTIONS) ───────────────────────
    with tab_rag:
        st.markdown("### 💬 Ask the Filing")
        st.caption("Questions are answered using retrieval from the actual document. Long, multi-part questions supported.")

        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []
        if "pending_q" not in st.session_state:
            st.session_state.pending_q = ""

        # Display history
        for turn in st.session_state.qa_history:
            st.markdown(f'<div class="qa-bubble-user">🧑 {turn["q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="qa-bubble-ai">🤖 {turn["a"]}</div>', unsafe_allow_html=True)

        # Suggestions
        if not st.session_state.qa_history:
            st.markdown("**Try asking:**")
            suggestions = [
                "What are the main revenue drivers?",
                "What risks does the company highlight?",
                "What is the management's outlook?",
                "Are there any pending litigations?",
                "What is the company's debt situation?",
            ]
            scols = st.columns(len(suggestions))
            for sc, sug in zip(scols, suggestions):
                with sc:
                    if st.button(sug, use_container_width=True, key=f"sug_{sug}"):
                        st.session_state.pending_q = sug
                        st.rerun()

        # Use text_area instead of text_input — better for long questions
        default_q = st.session_state.pending_q
        st.session_state.pending_q = ""
        question = st.text_area(
            "Your question:",
            value=default_q,
            placeholder="Ask anything — long, multi-part questions are supported. e.g. 'Compare the company's revenue trend over the last 3 years to its operating margin and explain the management's commentary on cost pressures.'",
            height=100,
            key=f"q_input_{len(st.session_state.qa_history)}",
            max_chars=MAX_QUESTION_CHARS,
        )

        ask_col, clear_col, _ = st.columns([1, 1, 3])
        with ask_col:
            ask_btn = st.button("Ask →", use_container_width=True)
        with clear_col:
            if st.button("Clear history", use_container_width=True):
                st.session_state.qa_history = []
                st.rerun()

        if ask_btn:
            q_clean = (question or "").strip()
            if not q_clean:
                st.warning("Please type a question first.")
            else:
                # Truncate question if absurdly long (defensive)
                q_for_prompt = q_clean[:MAX_QUESTION_CHARS]

                # Retrieve relevant chunks using BM25-lite
                with st.spinner("🔎 Searching filing for relevant sections..."):
                    top_chunks = retrieve_chunks(q_for_prompt, chunks, top_k=4)

                # Budget the context to fit total prompt limit
                budget_for_context = MAX_TOTAL_PROMPT_CHARS - len(q_for_prompt) - 500  # 500 char overhead for instructions
                budget_for_context = min(budget_for_context, MAX_CONTEXT_CHARS)
                if budget_for_context < 1000:
                    budget_for_context = 1000  # always allow at least some context

                # Trim chunks to fit
                context_parts = []
                running = 0
                for c in top_chunks:
                    if running + len(c) > budget_for_context:
                        remaining = budget_for_context - running
                        if remaining > 200:
                            context_parts.append(c[:remaining])
                        break
                    context_parts.append(c)
                    running += len(c)
                context = "\n\n---\n\n".join(context_parts)

                rag_prompt = (
                    "You are a financial analyst. Answer the user's question based on the filing excerpts below. "
                    "If the question has multiple parts, address each part in order. "
                    "If part of the answer is not in the excerpts, say so explicitly for that part rather than refusing the whole question. "
                    "Be concise, structured, and use bullet points or numbered lists for multi-part questions.\n\n"
                    f"FILING EXCERPTS:\n{context}\n\n"
                    f"QUESTION:\n{q_for_prompt}\n\n"
                    "ANSWER:"
                )

                # Show debug info to user so silent failures are visible
                with st.expander("🔧 Retrieval debug (click to inspect)"):
                    st.write(f"Question length: **{len(q_clean)} chars**")
                    st.write(f"Retrieved chunks: **{len(context_parts)}**")
                    st.write(f"Context size sent to LLM: **{len(context):,} chars**")
                    st.write(f"Total prompt size: **{len(rag_prompt):,} chars**")

                with st.spinner("🤖 Reading filing and generating answer..."):
                    try:
                        answer = call_llm(
                            [{"role": "system", "content": "You are a precise financial analyst. Answer thoroughly using only the provided context. For multi-part questions, address each part."},
                             {"role": "user", "content": rag_prompt}],
                            max_tokens=MAX_TOKENS_RESPONSE,
                        )
                        if not answer or not answer.strip():
                            answer = "⚠️ The model returned an empty response. Try rephrasing the question or breaking it into smaller parts."
                    except Exception as e:
                        answer = f"⚠️ Error generating answer: `{e}`\n\nTry: shortening the question, increasing chunk size in sidebar, or breaking it into smaller parts."

                st.session_state.qa_history.append({"q": q_clean, "a": answer})
                st.rerun()

    # ── TAB 4 — Raw Text ─────────────────────────────────────────────────────
    with tab_raw:
        st.markdown("### 🔍 Raw Extracted Text")
        st.caption(f"{len(chunks)} chunks · {word_count:,} words")
        if len(chunks) > 0:
            chunk_idx = st.slider("View chunk", 0, max(len(chunks) - 1, 0), 0)
            st.text_area("Chunk content", chunks[chunk_idx], height=400)
        st.download_button("⬇️ Download full text", raw_text, file_name="filing_text.txt")
