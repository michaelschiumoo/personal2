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