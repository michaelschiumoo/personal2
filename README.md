AI Synthesis Agent

AI Synthesis Agent turns 3–5 text sources into an extractive, traceable executive brief. It is extractive-only (no rewritten or invented sentences). Every claim is copied directly from a source sentence and includes a source_id + sentence index for traceability.

What it does

Input: a research question + 3–5 sources of text

Output: a report (Markdown or JSON) with:

Executive Summary (extractive)

Themes (clustered claim groups)

Evidence list for each theme (each item cites source + sentence index)

Uncertainties & gaps (low-support / assumption-heavy flags)

Next questions to reduce uncertainty

Audit metadata (config + similarity stats + agent decisions)

Hard guarantees

Extractive only: no paraphrasing, no “AI-written” claims

Traceability: every claim cites (source_id, sent_idx)

Deterministic fallbacks: avoids hard dependency on heavy ML libraries; will fall back to simpler vectorization/clustering when needed

How it works (high-level)

Split each source into sentences

Retrieve Top-K sentences per source most relevant to the question (default K=2)

Treat each selected sentence as a claim

Cluster claims into themes (target 3–5 themes)

Optional conservative post-merge (token overlap only; never forced)

If still too many themes, DISPLAY COMPACTION buckets leftovers into “other” (presentation only, not a semantic merge)

Emit Markdown or JSON plus an audit trail

Install / Requirements

Python 3.10+ recommended

No required ML dependencies

Optional: sentence-transformers and/or scikit-learn can improve embedding quality, but the program runs without them

CLI

Demo (no files needed)
python main.py demo --format markdown

Run on your own sources (file)
python main.py run --question "What factors affect AI adoption in organizations?" --sources sources.json --format markdown

Run on your own sources (stdin)
cat sources.json | python main.py run --question "..." --sources - --format json

Input format (sources.json)
Provide 3–5 sources in a JSON array:

[
{ "id": "source1", "text": "Text for source 1", "meta": { "url": "https://example.com
" } },
{ "id": "source2", "text": "Text for source 2" },
{ "id": "source3", "text": "Text for source 3" }
]

Notes / Common warning
You may see a warning like:
Low similarity for backend=pure-tfidf ... clustering fallback -> hashed-bow

This means the initial similarity scores were near-zero (common on tiny datasets), so the agent switched to a deterministic fallback backend for clustering. The report is still valid; it’s just being honest about the fallback.
