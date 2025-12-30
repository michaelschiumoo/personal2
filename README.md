# AI Synthesis Agent (grounded, traceable, agent-like)

A small CLI tool that produces a **grounded, traceable synthesis** from a tiny set of sources using an **extractive-only** approach (no invented sentences). It emphasizes auditability: every extracted claim is tied back to a `source_id` plus sentence index and clause index.

---

## CLI

### Demo (no files needed)

```bash
python main.py demo --format markdown

Run on your own sources

python main.py run --question "..." --sources sources.json --format json

Read sources JSON from stdin:

python main.py run --question "..." --sources - --format markdown

CLI arguments
	•	demo
	•	--format : json or markdown (default: markdown)
	•	--out : output path or - for stdout (default: -)
	•	run
	•	--question : research question (required)
	•	--sources : path to sources.json or - for stdin (required)
	•	--format : json or markdown (default: json)
	•	--out : output path or - for stdout (default: -)

If no arguments are provided, the program runs demo by default.

⸻

Input format (sources.json)

sources.json must be a JSON array of 3–5 items:

[
  {"id": "source1", "text": "....", "meta": {"url": "..."} },
  {"id": "source2", "text": "...."}
]

Each item:
	•	id (required): string identifier
	•	text (required): source text
	•	meta (optional): arbitrary object (e.g., { "url": "..." })

Notes:
	•	The tool enforces 3–5 sources.
	•	Empty id or text is rejected.

⸻

Design goals
	•	Extractive only (no invented sentences)
	•	Traceability: every claim cites source_id + sentence index (+ optional clause index)
	•	Deterministic fallback behavior (no hard dependency on heavy ML libs)
	•	Readable executive output on tiny datasets
	•	clustering + optional conservative post-merge
	•	then DISPLAY COMPACTION: bucket leftovers into "other" (NOT a semantic merge)

⸻

What it does

High-level flow:
	1.	Retrieve top-K relevant sentences per source (dependency-free TF-IDF retrieval).
	2.	Extract claims from retrieved sentences:
	•	split sentences into clauses (extractive)
	•	classify claim kind: fact | opinion | assumption | question
	•	infer directional stance: up | down | neutral | unknown
	3.	Embed / vectorize claims for clustering using the best available backend.
	4.	Cluster claims into themes with iterative threshold search to target an executive-friendly theme count.
	5.	Post-merge themes conservatively (token overlap; never forced).
	6.	Display-compaction: if too many themes remain, bucket extras into "other" (presentation only).
	7.	Detect agreements and potential contradictions (high similarity + direction cues).
	8.	Emit uncertainties, next questions, and an audit section.

⸻

Vector backends (auto-selected)

The tool will try (in order):
	1.	sentence-transformers backend (all-MiniLM-L6-v2) if installed
	2.	sklearn TF-IDF backend if installed
	3.	dependency-free pure TF-IDF fallback (always available)

For clustering, if similarities are extremely low under the chosen embedding backend (both p90 and max below a floor), it falls back to a hashed bag-of-words vectorization for clustering.

⸻

Clustering
	•	If scikit-learn is available, it uses AgglomerativeClustering with cosine distance.
	•	Otherwise it uses a greedy centroid-based clustering fallback.
	•	The agent iteratively adjusts a similarity threshold to aim for a theme count in the configured target range.

⸻

Theme labels and summaries
	•	Theme labels come from token-level TF-IDF within each theme (stable, dependency-free).
	•	Theme summaries are extractive: central claims in the theme, cited with (source=... sent=... clause=...).

⸻

Agreements and contradictions
	•	Agreements: claim pairs with cosine similarity ≥ agreement_threshold and not opposite direction.
	•	Contradictions: claim pairs with cosine similarity ≥ contradiction_threshold and opposite direction (up vs down).

These are heuristic “signals” meant to guide review.

⸻

Uncertainties and next questions

The tool flags themes that look weak or risky, including:
	•	low support (few claims)
	•	low source diversity (too few unique sources)
	•	dominated by opinions/assumptions
	•	internal inconsistency (contradiction-heavy)

It then generates follow-up questions to reduce uncertainty.

⸻

Output formats

JSON (--format json)

Produces a structured object including:
	•	question, backend, cluster_backend
	•	agent_plan, agent_decisions
	•	retrieval_trace, iteration_trace
	•	themes, key_claims_by_theme
	•	agreements, contradictions
	•	uncertainties, next_questions
	•	audit, scores

Markdown (--format markdown)

Produces a report including:
	•	Executive Summary (extractive)
	•	Agent Plan / Decisions
	•	Retrieval Trace / Iteration Trace
	•	Themes with evidence
	•	Agreements / Contradictions
	•	Uncertainties & Gaps / Next Questions
	•	Scores
	•	Audit (JSON block)

⸻

Logging

Logging is controlled by LOGLEVEL (default: INFO):

LOGLEVEL=DEBUG python main.py demo --format markdown


⸻

Demo data

The built-in demo uses five short sources and runs with the question:

What factors affect AI adoption in organizations?

Run it with:

python main.py demo --format markdown


⸻

Notes / constraints
	•	The system enforces 3–5 sources.
	•	The synthesis is extractive only by design.
	•	Theme compaction into "other" is explicitly presentation-only (not a semantic merge).

⸻


