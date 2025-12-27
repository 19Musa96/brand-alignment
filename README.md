# Celebrity–Brand Alignment AI (PoC)

A small, explainable Proof of Concept that compares two well-known entities (celebrity, brand, or public figure) using Wikipedia-only data and simple semantic similarity (to be added later).

## What This PoC Does
- Uses Wikipedia as the only data source (via official API summary endpoint).
- Fetches summaries for a celebrity and a brand, then computes semantic alignment via Gemini embeddings.
- Generates 3–5 concise explanation bullets using Gemini Pro based only on those summaries.

## What This PoC Does NOT Do
- No authentication, caching, scaling, or performance optimizations.
- No scraping beyond the official Wikipedia API.
- No fine-tuning or external datasets.
- No production hardening, monitoring, or advanced error handling.

## Tech Stack
- Python 3.11
- Streamlit (UI)
- Google Gemini API: `text-embedding-004` for embeddings, `gemini-pro` for explanations
- `requests` for Wikipedia REST calls
- `python-dotenv` for environment variables

## Alignment Scoring (High Level)
1) Fetch Wikipedia summaries for both entities via the REST summary endpoint.
2) Generate embeddings with Gemini `text-embedding-004`.
3) Compute cosine similarity and map from [-1, 1] to a 0–100 alignment score.
4) Ask Gemini Pro to produce 3–5 factual bullets using only the fetched summaries.

## Project Layout
```
app.py
services/
  ├─ wikipedia_service.py
  ├─ embedding_service.py
  └─ explanation_service.py
utils/
  └─ text_utils.py
README.md
requirements.txt
.env.example
```

## Run Locally (PoC)
1. Ensure Python 3.11 is installed.
2. Create and activate a virtual environment:
   - Windows (PowerShell):
     ```powershell
     py -3.11 -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy .env example and update values:
   ```bash
   copy .env.example .env
   ```
5. Set environment variables (inside the shell or .env):
   - `GEMINI_API_KEY` (required for embeddings and explanations)
   - `WIKIPEDIA_USER_AGENT` (required for Wikipedia REST requests)
6. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deploy (Streamlit Cloud)
1. Push this repo to your Git host.
2. Create a new Streamlit Cloud app pointing to `app.py`.
3. Set secrets/env vars in the Streamlit Cloud settings:
   - `GEMINI_API_KEY`
   - `WIKIPEDIA_USER_AGENT`
4. Deploy; the app is single-page and requires no extra services.

## Notes
- This is a Proof of Concept (PoC), not a production system.
- Single-model, no batching/caching, minimal error handling by design.
