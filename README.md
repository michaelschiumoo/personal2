# AI Synthesis Agent (grounded, traceable, agent-like)

A small CLI tool that produces a **grounded, traceable synthesis** from a tiny set of sources using an **extractive-only** approach (no invented sentences).  
It emphasizes auditability: every extracted claim is tied back to a `source_id` plus sentence index and clause index.

---

## CLI

### Demo (no files needed)

```bash
python main.py demo --format markdown

Run on your own sources

python main.py run --question "..." --sources sources.json --format json

Read sources JSON from stdin

python main.py run --question "..." --sources - --format markdown


⸻

CLI arguments

demo
	•	--format: json or markdown (default: markdown)
	•	--out: output path or - for stdout (default: -)

run
	•	--question: research question (required)
	•	--sources: path to sources.json or - for stdin (required)
	•	--format: json or markdown (default: json)
	•	--out: output path or - for stdout (default: -)

If no arguments are provided, the program runs the built-in demo.

⸻

Input format (sources.json)

sources.json must be a JSON array of 3–5 sources:

[
  {"id": "source1", "text": "....", "meta": {"url": "..."}},
  {"id": "source2", "text": "...."}
]

Each item:
	•	id (required): string identifier
	•	text (required): source text
	•	meta (optional): arbitrary object (e.g., { "url": "..." })

Notes:
	•	The tool enforces 3–5 sources
	•	Empty id or text is rejected

⸻

Design goals
	•	Extractive only (no invented sentences)
	•	Traceability: every claim cites source_id + sentence index (+ optional clause index)
	•	Deterministic fallback behavior (no hard dependency on heavy ML libs)
	•	Readable executive output on tiny datasets:
	•	clustering + optional conservative post-merge
	•	then DISPLAY COMPACTION: bucket leftovers into "other" (NOT a semantic merge)

Recent upgrades (priority fixes):
	•	Higher evidence coverage (TopK default higher) + unique-sources-per-theme uncertainty
	•	More robust similarity fallback decision (p90 AND max must be low)
	•	Clause-level claim extraction (split sentences into atomic-ish clauses)
	•	Improved contradiction detection via direction/stance cues (not just negation count)
	•	Better scoring (coverage + multi-source support)
	•	More stable theme labels (token TF-IDF within cluster), no crude plural chopping

⸻

What it does

High-level flow:
	1.	Retrieve top-K relevant sentences per source (TF-IDF retrieval)
	2.	Extract claims from retrieved sentences:
	•	split sentences into clauses
	•	classify claim kind: fact, opinion, assumption, question
	•	infer directional stance: up, down, neutral, unknown
	3.	Embed / vectorize claims
	4.	Cluster claims into themes
	5.	Post-merge themes conservatively (token overlap; never forced)
	6.	Display-compaction: bucket excess themes into "other" (presentation-only)
	7.	Detect agreements / contradictions (similarity + direction cues)
	8.	Emit uncertainties, next questions, audit, scores

⸻

Vector backends (auto-selected)
	1.	sentence-transformers backend (if installed)
	2.	sklearn TF-IDF backend (if installed)
	3.	dependency-free pure TF-IDF fallback (always available)

For clustering, if similarities are extremely low (both p90 and max below a floor), clustering falls back to a hashed bag-of-words backend.

⸻

Clustering
	•	If scikit-learn is available, uses agglomerative clustering (cosine distance)
	•	Otherwise uses a greedy centroid clustering fallback
	•	Iteratively adjusts similarity threshold to target an executive-friendly theme count

⸻

Theme labels and summaries
	•	Theme labels come from token TF-IDF keywords within each theme
	•	Theme summaries are extractive: central claims cited with (source=... sent=... clause=...)

⸻

Agreements and contradictions
	•	Agreements: high similarity + not opposite direction
	•	Contradictions: high similarity + opposite direction (up vs down)

These are heuristic signals meant to guide review.

⸻

Uncertainties and next questions

The tool flags themes that look weak:
	•	low support (few claims)
	•	low source diversity (few unique sources)
	•	dominated by opinions/assumptions
	•	internal inconsistency (contradiction-heavy)

Then it generates follow-up questions to reduce uncertainty.

⸻

Output formats

JSON (--format json)

Structured output including:
	•	themes, evidence, agreements/contradictions, uncertainties, next questions
	•	audit trail (backend choice, thresholds, similarity stats)
	•	scores (coverage, consistency, multi-source support, etc.)

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

The built-in demo uses five short sources and the question:

What factors affect AI adoption in organizations?

Run it with:

python main.py demo --format markdown


⸻

Notes / constraints
	•	The system enforces 3–5 sources
	•	The synthesis is extractive-only by design
	•	Theme compaction into "other" is presentation-only (not a semantic merge)

