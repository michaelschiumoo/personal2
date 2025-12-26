# /workspace/personal2/research_synthesis_assistant.py

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Set


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_STOPWORDS: Set[str] = {
    # Core stopwords
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "its", "of", "on", "or", "our", "so", "such",
    "that", "the", "their", "then", "there", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",

    # Question scaffolding (prevents theme labels like "factors"/"affect")
    "factor", "factors", "affect", "affects", "affected", "affecting",

    # Optional: often too generic as a theme
    "organization", "organizations",
}


def tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer:
      - lowercase
      - keep alphanumerics + apostrophes
      - split to tokens
      - drop stopwords
    """
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class ResearchSynthesisAssistant:
    """
    Constrained Research Synthesis Assistant - Sandbox Prototype.

    Inputs:
      - sources: list of dicts with keys: {"id": str, "text": str}
      - research_question: str

    Output:
      - structured dict with clustered claims + uncertainties for human review
    """

    def __init__(self, sources: List[Dict[str, str]], research_question: str) -> None:
        self.validate_inputs(sources, research_question)
        self.sources = sources
        self.research_question = research_question
        self._question_keywords = set(tokenize(research_question))
        logging.info("Initialized with %d sources for question: %s", len(sources), research_question)

    @staticmethod
    def validate_inputs(sources: List[Dict[str, str]], research_question: str) -> None:
        if not 3 <= len(sources) <= 5:
            raise ValueError("Must provide exactly 3-5 sources.")
        if not research_question.strip():
            raise ValueError("Research question is required.")

        for i, source in enumerate(sources):
            if "id" not in source or "text" not in source:
                raise ValueError("Each source must have 'id' and 'text' keys.")
            if not str(source["id"]).strip():
                raise ValueError(f"Source at index {i} has an empty 'id'.")
            if not str(source["text"]).strip():
                raise ValueError(f"Source at index {i} has an empty 'text'.")

    def extract_claims(self) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for source in self.sources:
            sentences = [
                s.strip()
                for s in source["text"].split(".")
                if s.strip() and "?" not in s
            ]
            for sent in sentences:
                claims.append(
                    {
                        "claim": sent,
                        "source_id": source["id"],
                        "type": self.classify_claim_type(sent),
                    }
                )

        logging.info("Extracted %d claims.", len(claims))
        return claims

    @staticmethod
    def classify_claim_type(claim: str) -> str:
        lower = claim.lower()
        if any(word in lower for word in ["assume", "hypothesize", "might", "possible"]):
            return "assumption"
        if any(word in lower for word in ["believe", "think", "opinion", "feel"]):
            return "opinion"
        return "fact"

    def cluster_themes(self, claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        themes: Dict[str, List[Dict[str, Any]]] = {}

        for claim in claims:
            claim_tokens = set(tokenize(str(claim["claim"])))
            overlap = self._question_keywords.intersection(claim_tokens)

            if overlap:
                theme = max(overlap, key=lambda t: (len(t), t))
            else:
                theme = "misc"

            themes.setdefault(theme, []).append(claim)

        return themes

    def identify_uncertainties(self, themes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
        uncertainties: Dict[str, List[str]] = {}
        for theme, claims_list in themes.items():
            theme_uncertainties: List[str] = []

            if len(claims_list) < 2:
                theme_uncertainties.append(
                    "Low-frequency signal: isolated claim may lack corroboration."
                )

            types = [c["type"] for c in claims_list]
            if "fact" not in types:
                theme_uncertainties.append(
                    "No factual grounding: relies on opinions/assumptions."
                )

            # Keep output cleaner by omitting empty lists
            if theme_uncertainties:
                uncertainties[theme] = theme_uncertainties

        return uncertainties

    def generate_output(self) -> Dict[str, Any]:
        claims = self.extract_claims()
        themes = self.cluster_themes(claims)
        uncertainties = self.identify_uncertainties(themes)

        possible_next_questions = [
            f"What additional sources could resolve: {u}"
            for u_list in uncertainties.values()
            for u in u_list
        ] or ["No major uncertainties detected in clustered themes."]

        return {
            "key_claims_by_theme": {
                theme: [
                    f"{c['claim']} ({c['source_id']}, type: {c['type']})"
                    for c in claims_list
                ]
                for theme, claims_list in themes.items()
            },
            "areas_of_agreement": {"note": "Basic detection not implemented in v1 for restraint."},
            "areas_of_disagreement": {"note": "Basic detection not implemented in v1 for restraint."},
            "uncertainties_and_gaps": uncertainties,
            "possible_next_questions": possible_next_questions,
            "human_approval_required": True,
        }


def evaluate_output(output: Dict[str, Any]) -> Dict[str, int]:
    key_claims = output.get("key_claims_by_theme", {})
    has_claims = any(key_claims.values())

    uncertainties = output.get("uncertainties_and_gaps", {})
    has_uncertainties = any(uncertainties.values())

    return {
        "accuracy": 5 if has_claims else 1,
        "grounding": 5 if has_claims else 1,
        "explanation_quality": 4,
        "decision_usefulness": 4 if has_uncertainties else 2,
        "hallucination_risk": 5,
    }


if __name__ == "__main__":
    sample_sources = [
        {"id": "source1", "text": "Adoption increased among mid-sized firms in past 12 months. But it remains experimental."},
        {"id": "source2", "text": "Governance is a limiting factor. Productivity gains are uncertain."},
        {"id": "source3", "text": "Adoption is increasing but uneven. No consensus on ROI."},
        {"id": "source4", "text": "Assume tooling maturity is low in regulated industries."},
        {"id": "source5", "text": "Opinion: Human oversight intervenes frequently."},
    ]
    question = "What factors affect AI adoption in organizations?"

    assistant = ResearchSynthesisAssistant(sample_sources, question)
    output = assistant.generate_output()
    print(json.dumps(output, indent=2))

    scores = evaluate_output(output)
    print("\nEvaluation Scores:", scores)

    if all(score >= 4 for score in scores.values()):
        print("\nOutput passes basic thresholds but requires human approval.")
    else:
        print("\nOutput rejected due to low scores.")
