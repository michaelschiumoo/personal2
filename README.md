# AI Synthesis Agent (Grounded, Traceable, Extractive)

A command-line agent that synthesizes grounded, extractive summaries from 3–5 textual sources.  
It produces transparent, auditable results — every claim is directly cited with source ID and sentence indices.

---

## 🚀 Features

- **Extractive only:** all output text comes directly from source sentences.
- **Traceable:** every claim includes `source_id`, `sent_idx`, `clause_idx`.
- **Deterministic:** no stochastic ML dependencies required.
- **Lightweight:** pure-Python fallback for all components.
- **Readable output:** executive-friendly themes + compact display mode.

---

## 🧩 Requirements

- Python 3.8 or newer (tested up to 3.12)
- Optional (auto-used if installed):
  - `sentence-transformers`
  - `scikit-learn`
  - `numpy`

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/ai-synthesis-agent.git
cd ai-synthesis-agent
python -m pip install -U pip
# Optional for higher-quality embeddings:
python -m pip install sentence-transformers scikit-learn numpy

🧠 Quick Start

Run the built-in demo

python main.py demo --format markdown

or in JSON:

python main.py demo --format json


⸻

📚 Use on your own sources

python main.py run --question "What factors affect AI adoption in organizations?" --sources sources.json --format markdown

Read from stdin:

cat sources.json | python main.py run --question "..." --sources - --format json

Arguments

Flag	Description	Default
--question	Research question (required)	—
--sources	Path to sources.json or - for stdin	—
--format	json or markdown	json
--out	Output path or - for stdout	-


⸻

🧾 Input Format (sources.json)

[
  {"id": "source1", "text": "Adoption increased among mid-sized firms."},
  {"id": "source2", "text": "Governance is a limiting factor."},
  {"id": "source3", "text": "Assume tooling maturity is low in regulated industries."}
]

Each item must include:
	•	id — unique string identifier
	•	text — plain text content
	•	meta — optional dictionary (e.g., { "url": "https://..." })


🧮 How It Works
	1.	Sentence Retrieval
Pure TF-IDF (PureTfidfBackend) retrieves top-K relevant sentences per source.
	2.	Clause Extraction
Each retrieved sentence is split into extractive clauses using punctuation and discourse markers.
	3.	Claim Classification
Each clause is labeled as fact, opinion, assumption, or question.
	4.	Vectorization
Backends auto-select in order:
SentenceTransformerBackend → SklearnTfidfBackend → PureTfidfBackend.
	5.	Clustering
Agglomerative or fallback greedy clustering groups similar clauses into themes.
	6.	Labeling & Summaries
Each theme gets keyword labels and extractive summary bullets.
	7.	Post-Merge & Compaction
Optionally merges overlapping themes, then buckets leftovers into "other".
	8.	Agreements & Contradictions
Detects directional consistency across sources.
	9.	Uncertainties & Next Questions
Flags themes with low support or internal inconsistency.


🧾 Output Example (Markdown)

python main.py demo --format markdown

Example snippet:

# AI Synthesis Agent Report

**Question:** What factors affect AI adoption in organizations?  
**Backend:** `pure-tfidf`  
**Cluster Backend:** `pure-tfidf`  
**Claims:** 10  
**Themes:** 3

The markdown report includes:
	•	Executive summary
	•	Agent plan
	•	Theme breakdowns with extractive claims
	•	Agreements and contradictions
	•	Uncertainties & next questions
	•	Audit block with backend decisions and thresholds


🧰 JSON Output

Example run:

python main.py demo --format json | jq .

Top-level keys:

{
  "question": "...",
  "backend": "pure-tfidf",
  "themes": [...],
  "agreements": [...],
  "contradictions": [...],
  "uncertainties": [...],
  "next_questions": [...],
  "audit": {...},
  "scores": {...}
}



⚙️ Configuration (AgentConfig)

Defined in main.py.
Important defaults:

Setting	Default	Description
topk_sentences_per_source	6	Sentences per source
target_min_themes	3	Min desired themes
target_max_themes	5	Max desired themes
agreement_threshold	0.82	Cosine sim for agreement
contradiction_threshold	0.82	Cosine sim for contradiction
low_similarity_floor	0.20	Triggers backend fallback
compact_excess_themes	True	Buckets extras into "other"


🧩 Logging

Use LOGLEVEL to control verbosity:

LOGLEVEL=DEBUG python main.py demo --format json


📊 Outputs & Scores

Each synthesis result includes transparent scores:
	•	traceability
	•	consistency
	•	coverage
	•	multi_source_theme_fraction
	•	usefulness
	•	hallucination_risk_control

⸻

🧾 License

MIT (add or modify as needed)

