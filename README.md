# FilingIQ — Regulatory Intelligence Platform

An AI-powered Streamlit app that turns regulatory filings (SEC 10-K / 10-Q, SEBI annual reports, earnings releases) into structured intelligence. Upload a filing PDF and get a research-grade summary, financial visualizations, and a retrieval-grounded Q&A chat.

Built during a TCS AI hackathon.

## Features

- **Summary Report** — domain-specific extraction of financials, risk factors, management highlights, and sentiment, plus an institutional-style research note.
- **Statistics & Charts** — revenue, net income, operating margin, EPS, a risk-factor radar, and a keyword frequency chart.
- **Ask the Filing** — RAG-powered Q&A using a lightweight BM25-style retriever, with support for long, multi-part questions and a token/character budget so prompts stay within limits.
- **Raw Text** — browse extracted chunks and download the full parsed text.

## Tech stack

Streamlit · PyMuPDF · pandas / numpy · Plotly · LangChain (OpenAI-compatible client) · httpx

## Getting started

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your LLM credentials. Copy the example env file and fill in your own values:

   ```bash
   cp .env.example .env
   # then edit .env
   ```

   The app reads three variables: `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL_ENDPOINT`. The app works with any OpenAI-compatible endpoint.

3. Run it:

   ```bash
   streamlit run app.py
   ```

## Configuration

| Variable | Description |
| --- | --- |
| `LLM_API_KEY` | API key for your LLM endpoint (required) |
| `LLM_BASE_URL` | Base URL of an OpenAI-compatible endpoint |
| `LLM_MODEL_ENDPOINT` | Model identifier to call |

Credentials are never hardcoded — they are read from the environment. The `.env` file is gitignored.

## Notes

- The financial charts use modeled/sample time-series for demonstration. For production use, wire them to real extracted figures or a market-data source.
- `httpx` is configured with `verify=False` to accommodate the original internal endpoint. Remove that flag (or supply a proper CA bundle) for any public deployment.

## License

MIT
