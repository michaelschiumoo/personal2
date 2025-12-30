# AI Synthesis Agent (grounded, traceable, agent-like)

CLI:

- Demo (no files needed):
  - `python main.py demo --format markdown`

- Run on your own sources:
  - `python main.py run --question "..." --sources sources.json --format json`
  - `python main.py run --question "..." --sources - --format markdown`  (read sources JSON from stdin)

`sources.json` format:

```json
[
  {"id": "source1", "text": "....", "meta": {"url": "..."} },
  {"id": "source2", "text": "...."}
]

