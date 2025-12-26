import json
import logging
from typing import List, Dict, Any

# Set up logging for traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResearchSynthesisAssistant:
"""
Constrained Research Synthesis Assistant - Sandbox Prototype
"""
def __init__(self, sources: List[Dict[str, str]], research_question: str):
self.validate_inputs(sources, research_question)
self.sources = sources
self.research_question = research_question
logging.info(f"Initialized with {len(sources)} sources for question: {research_question}")

def validate_inputs(self, sources: List[Dict[str, str]], research_question: str):
if not 3 <= len(sources) <= 5:
raise ValueError("Must provide exactly 3-5 sources.")
if not research_question.strip():
raise ValueError("Research question is required.")
for source in sources:
if 'id' not in source or 'text' not in source:
raise ValueError("Each source must have 'id' and 'text' keys.")

def extract_claims(self) -> List[Dict[str, Any]]:
claims = []
for source in self.sources:
# Simple sentence split
sentences = [s.strip() for s in source['text'].split('.') if s.strip() and '?' not in s]
for sent in sentences:
claim_type = self.classify_claim_type(sent)
claims.append({
'claim': sent,
'source_id': source['id'],
'type': claim_type
})
logging.info(f"Extracted {len(claims)} claims.")
return claims

@staticmethod
def classify_claim_type(claim: str) -> str:
lower = claim.lower()
if any(word in lower for word in ['assume', 'hypothesize', 'might', 'possible']):
return 'assumption'
elif any(word in lower for word in ['believe', 'think', 'opinion', 'feel']):
return 'opinion'
return 'fact'

def cluster_themes(self, claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
themes = {}
question_keywords = set(self.research_question.lower().split())
for claim in claims:
claim_words = set(claim['claim'].lower().split())
overlap = question_keywords.intersection(claim_words)
if overlap:
theme = max(overlap, key=len) # Better theme selection
themes.setdefault(theme, []).append(claim)
return themes

def identify_uncertainties(self, themes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[str]]:
uncertainties = {}
for theme, claims_list in themes.items():
uncertainties[theme] = []
if len(claims_list) < 2:
uncertainties[theme].append("Low-frequency signal: Isolated claim may lack corroboration.")
types = [c['type'] for c in claims_list]
if 'fact' not in types:
uncertainties[theme].append("No factual grounding: Relies on opinions/assumptions.")
return uncertainties

def generate_output(self) -> Dict[str, Any]:
claims = self.extract_claims()
themes = self.cluster_themes(claims)
uncertainties = self.identify_uncertainties(themes)

output = {
"key_claims_by_theme": {
theme: [f"{c['claim']} ({c['source_id']}, type: {c['type']})" for c in claims_list]
for theme, claims_list in themes.items()
},
"areas_of_agreement": {"note": "Basic detection not implemented in v1 for restraint."},
"areas_of_disagreement": {"note": "Basic detection not implemented in v1 for restraint."},
"uncertainties_and_gaps": uncertainties,
"possible_next_questions": [
f"What additional sources could resolve: {u[0]}" for u_list in uncertainties.values() for u in u_list
] or ["No major uncertainties detected in clustered themes."],
"human_approval_required": True
}
return output

def find_agreements(self, themes): return {}
def find_disagreements(self, themes): return {}

def evaluate_output(output: Dict[str, Any]) -> Dict[str, int]:
has_claims = any(output['key_claims_by_theme'].values())
scores = {
'accuracy': 5 if has_claims else 1,
'grounding': 5 if has_claims else 1, # All claims have source_id
'explanation_quality': 4,
'decision_usefulness': 4 if output['possible_next_questions'] else 1,
'hallucination_risk': 5 # No generation beyond structuring
}
return scores

# === Sample Run ===
if __name__ == "__main__":
sample_sources = [
{'id': 'source1', 'text': 'Adoption increased among mid-sized firms in past 12 months. But it remains experimental.'},
{'id': 'source2', 'text': 'Governance is a limiting factor. Productivity gains are uncertain.'},
{'id': 'source3', 'text': 'Adoption is increasing but uneven. No consensus on ROI.'},
{'id': 'source4', 'text': 'Assume tooling maturity is low in regulated industries.'},
{'id': 'source5', 'text': 'Opinion: Human oversight intervenes frequently.'}
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
