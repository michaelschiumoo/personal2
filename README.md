# Constrained Research Synthesis Assistant

This is a sandboxed, non-production prototype implementing the "Research Synthesis Assistant" described in my proof-of-capability artifact.

## Purpose
Support human-led research synthesis by:
- Extracting and attributing concrete claims from a small set of provided sources (3–5 only)
- Clustering claims by relevance to a research question
- Explicitly flagging uncertainties, gaps, and potential contradictions
- Suggesting possible next research questions

The assistant **never** draws conclusions, makes decisions, or generates recommendations. All outputs require mandatory human review.

## Design Constraints (Enforced in Code)
- No external retrieval or internet access
- Fixed small input size (3–5 sources) to prevent false confidence from aggregation
- Strict source attribution for every claim
- No final framing or synthesis narrative
- Built-in input validation and logging for traceability
- Explicit `human_approval_required` flag in output

These constraints mitigate common failure modes: hallucinations, over-smoothing of conflicts, and premature closure.

## Failure Modes Addressed
- Hallucinations → forced grounding and attribution
- Bias toward verbose/recent sources → uniform extraction
- Missing high-impact isolated signals → explicit uncertainty flagging
- Human deference → banned conclusions and mandatory review

## Usage (Local Only)
```bash
python synthesis_assistant.py
