# AI Synthesis Agent

A small Python CLI that turns 3–5 text sources into an **extractive** executive brief.  
**Every claim is copied from your sources** and includes `source_id + sentence index (+ clause index)` for traceability.

## What it does

Given a research question and a small set of sources, it:

1. Splits sources into sentences, retrieves top-K sentences per source (deterministic).
2. Extracts **claims** (extractive; no new text invented).
3. Optionally splits long sentences into **clauses** so one sentence doesn’t become one oversized claim.
4. Clusters claims into **themes** (tries nicer embeddings if installed; otherwise uses dependency-free TF-IDF).
5. If similarities are too low for the main backend, it falls back to a simple **hashed bag-of-words** clustering backend.
6. Produces a Markdown or JSON report:
   - Executive summary (extractive)
   - Themes with evidence bullets (extractive)
   - Uncertainties + next questions
   - Audit + scoring

## Key guarantees

- **Extractive only**: output claims are taken verbatim from input text.
- **Traceable**: each claim includes `source`, `sent`, and optional `clause`.
- **Deterministic fallbacks**: runs without requiring heavy ML libraries.

## Install

Python 3.10+ recommended.

No required dependencies beyond the standard library.

Optional (improves similarity/cluster quality if available):
- `sentence-transformers`
- `scikit-learn`
- `numpy`

## CLI usage

### Demo (no files needed)

```bash
python main.py demo --format markdown

