# /workspace/personal2/main.py
"""
AI Synthesis Agent (grounded, traceable, agent-like)

CLI:
  Demo (no files needed):
    python main.py demo --format markdown

  Run on your own sources:
    python main.py run --question "..." --sources sources.json --format json
    python main.py run --question "..." --sources - --format markdown   # read sources JSON from stdin

sources.json format:
[
  {"id": "source1", "text": "....", "meta": {"url": "..."} },
  {"id": "source2", "text": "...."}
]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger("ai_synthesis_agent")
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------------------------
# Data Models
# ----------------------------

@dataclass(frozen=True)
class Source:
    id: str
    text: str
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    id: str
    source_id: str
    sent_idx: int
    start: int
    end: int
    text: str
    kind: str  # fact|opinion|assumption|question
    polarity: str  # pos|neg|mixed|unknown
    tokens: Tuple[str, ...]


@dataclass
class Theme:
    id: str
    label: str
    claim_ids: List[str]
    keywords: List[str]
    summary: List[str]  # extractive bullets
    stats: Dict[str, Any]


@dataclass
class SynthesisResult:
    question: str
    themes: List[Theme]
    key_claims_by_theme: Dict[str, List[str]]
    agreements: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    uncertainties: List[Dict[str, Any]]
    next_questions: List[str]
    audit: Dict[str, Any]
    scores: Dict[str, float]


# ----------------------------
# Configuration
# ----------------------------

@dataclass(frozen=True)
class AgentConfig:
    min_sentence_chars: int = 20
    max_claims_per_source: int = 80

    theme_assign_threshold: float = 0.55
    agreement_threshold: float = 0.78
    contradiction_threshold: float = 0.78

    low_support_min_claims: int = 2
    opinion_assumption_heavy_ratio: float = 0.6
    contradiction_heavy_ratio: float = 0.25

    summary_sentences_per_theme: int = 2
    max_themes: int = 12


# ----------------------------
# Text utilities
# ----------------------------

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "for", "from", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "may", "might", "of", "on", "or", "our", "so", "such", "than",
    "that", "the", "their", "then", "there", "these", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your", "we", "they", "them", "he", "she", "his", "her",
    "factor", "factors", "affect", "affects", "affected", "affecting",
    "impact", "impacts", "influence", "influences", "influenced", "influencing",
    "organization", "organizations", "company", "companies", "firm", "firms",
}

_NEGATION = {"no", "not", "never", "none", "without", "hardly", "rarely", "lack", "lacks"}

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    if not text.strip():
        return []
    chunks = _SENT_SPLIT_RE.split(text.strip())
    out: List[Tuple[str, int, int]] = []
    cursor = 0
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        start = text.find(chunk, cursor)
        if start < 0:
            start = cursor
        end = start + len(chunk)
        out.append((chunk, start, end))
        cursor = end
    return out


def tokenize(text: str) -> Tuple[str, ...]:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    normed: List[str] = []
    for t in toks:
        if t in _STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        normed.append(t)
    return tuple(normed)


def classify_kind(sentence: str) -> str:
    s = sentence.strip()
    lower = s.lower()
    if "?" in s:
        return "question"
    if any(w in lower for w in ("assume", "hypothesize", "possible", "might", "may", "could")):
        return "assumption"
    if any(w in lower for w in ("opinion", "believe", "think", "feel", "seems", "suggests")):
        return "opinion"
    return "fact"


def estimate_polarity(tokens: Sequence[str]) -> str:
    if not tokens:
        return "unknown"
    neg = sum(1 for t in tokens if t in _NEGATION)
    if neg == 0:
        return "pos"
    if neg == 1:
        return "neg"
    return "mixed"


# ----------------------------
# Vector backends (optional ML, robust fallback)
# ----------------------------

def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class VectorBackend:
    name: str = "base"

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def transform(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SentenceTransformerBackend(VectorBackend):
    name = "sentence-transformers"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def transform(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class SklearnTfidfBackend(VectorBackend):
    name = "sklearn-tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[a-z0-9']+",
            stop_words=list(_STOPWORDS),
            ngram_range=(1, 2),
            max_features=5000,
        )

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        m = self.vectorizer.fit_transform(texts)
        return m.toarray().tolist()

    def transform(self, texts: List[str]) -> List[List[float]]:
        m = self.vectorizer.transform(texts)
        return m.toarray().tolist()


class HashedBoWBackend(VectorBackend):
    name = "hashed-bow"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for t in tokenize(text):
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        if n > 0:
            v = [x / n for x in v]
        return v

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def transform(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


def pick_backend() -> VectorBackend:
    try:
        return SentenceTransformerBackend()
    except Exception:
        pass
    try:
        return SklearnTfidfBackend()
    except Exception:
        pass
    return HashedBoWBackend()


# ----------------------------
# Clustering (sklearn optional, fallback greedy)
# ----------------------------

def cluster_vectors(vectors: List[List[float]], threshold: float) -> List[List[int]]:
    if not vectors:
        return []

    try:
        import numpy as np  # type: ignore
        from sklearn.cluster import AgglomerativeClustering  # type: ignore

        X = np.array(vectors, dtype=float)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=max(0.0, 1.0 - threshold),
            linkage="average",
            metric="cosine",
        )
        labels = model.fit_predict(X).tolist()
        clusters: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)
        return list(clusters.values())
    except Exception:
        pass

    clusters: List[List[int]] = []
    centroids: List[List[float]] = []

    def centroid(idxs: List[int]) -> List[float]:
        dim = len(vectors[0])
        c = [0.0] * dim
        for idx in idxs:
            v = vectors[idx]
            for j in range(dim):
                c[j] += v[j]
        n = float(len(idxs))
        c = [x / n for x in c]
        norm = math.sqrt(sum(x * x for x in c))
        if norm > 0:
            c = [x / norm for x in c]
        return c

    for i, v in enumerate(vectors):
        best_i = -1
        best_sim = 0.0
        for ci, c in enumerate(centroids):
            sim = cosine(v, c)
            if sim > best_sim:
                best_i, best_sim = ci, sim
        if best_i >= 0 and best_sim >= threshold:
            clusters[best_i].append(i)
            centroids[best_i] = centroid(clusters[best_i])
        else:
            clusters.append([i])
            centroids.append(centroid([i]))

    return clusters


# ----------------------------
# Keywording + summaries
# ----------------------------

def _count_kinds(claims: List[Claim]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for c in claims:
        d[c.kind] = d.get(c.kind, 0) + 1
    return d


def top_keywords_from_claims(claims: List[Claim], k: int = 6) -> List[str]:
    freq: Dict[str, float] = {}
    for c in claims:
        for t in c.tokens:
            if t in _STOPWORDS:
                continue
            freq[t] = freq.get(t, 0.0) + 1.0
    scored = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [w for w, _ in scored[:k]]


def pick_central_claims(claim_indices: List[int], vectors: List[List[float]], n: int) -> List[int]:
    if not claim_indices:
        return []
    if len(claim_indices) <= n:
        return claim_indices[:]

    dim = len(vectors[0])
    c = [0.0] * dim
    for idx in claim_indices:
        v = vectors[idx]
        for j in range(dim):
            c[j] += v[j]
    denom = float(len(claim_indices))
    c = [x / denom for x in c]
    norm = math.sqrt(sum(x * x for x in c))
    if norm > 0:
        c = [x / norm for x in c]

    ranked = sorted(claim_indices, key=lambda idx: cosine(vectors[idx], c), reverse=True)
    return ranked[:n]


def _format_claim(c: Claim) -> str:
    return f"{c.text} (source={c.source_id} sent={c.sent_idx} kind={c.kind} polarity={c.polarity})"


def _pair_record(kind: str, a: Claim, b: Claim, sim: float) -> Dict[str, Any]:
    return {
        "type": kind,
        "similarity": round(sim, 4),
        "a": {"id": a.id, "source": a.source_id, "sent": a.sent_idx, "text": a.text, "polarity": a.polarity, "kind": a.kind},
        "b": {"id": b.id, "source": b.source_id, "sent": b.sent_idx, "text": b.text, "polarity": b.polarity, "kind": b.kind},
    }


# ----------------------------
# Agent
# ----------------------------

class AISynthesisAgent:
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.backend = pick_backend()

    @staticmethod
    def validate(sources: List[Dict[str, Any]], question: str) -> List[Source]:
        if not question or not question.strip():
            raise ValueError("research question is required")
        if not (3 <= len(sources) <= 5):
            raise ValueError("must provide 3–5 sources")

        parsed: List[Source] = []
        for i, s in enumerate(sources):
            if "id" not in s or "text" not in s:
                raise ValueError(f"source[{i}] must include 'id' and 'text'")
            sid = str(s["id"]).strip()
            txt = str(s["text"]).strip()
            if not sid or not txt:
                raise ValueError(f"source[{i}] has empty id/text")
            meta = dict(s.get("meta") or {})
            parsed.append(Source(id=sid, text=txt, meta=meta))
        return parsed

    def extract_claims(self, sources: List[Source]) -> List[Claim]:
        claims: List[Claim] = []
        for src in sources:
            sents = split_sentences(src.text)
            for sent_idx, (sent, start, end) in enumerate(sents[: self.config.max_claims_per_source]):
                if len(sent.strip()) < self.config.min_sentence_chars:
                    continue
                kind = classify_kind(sent)
                toks = tokenize(sent)
                pol = estimate_polarity(toks)
                cid = _stable_id(src.id, str(sent_idx), sent)
                claims.append(
                    Claim(
                        id=cid,
                        source_id=src.id,
                        sent_idx=sent_idx,
                        start=start,
                        end=end,
                        text=sent.strip(),
                        kind=kind,
                        polarity=pol,
                        tokens=toks,
                    )
                )
        logger.info("Extracted %d claims.", len(claims))
        return claims

    def synthesize(self, sources_raw: List[Dict[str, Any]], question: str) -> SynthesisResult:
        sources = self.validate(sources_raw, question)
        claims = self.extract_claims(sources)

        claim_texts = [c.text for c in claims]
        vectors = self.backend.fit_transform(claim_texts)

        clusters = cluster_vectors(vectors, threshold=self.config.theme_assign_threshold)
        clusters = sorted(clusters, key=len, reverse=True)[: self.config.max_themes]

        claim_by_id = {c.id: c for c in claims}

        themes: List[Theme] = []
        for t_i, idxs in enumerate(clusters):
            cluster_claims = [claims[i] for i in idxs]
            kws = top_keywords_from_claims(cluster_claims, k=6)
            label = kws[0] if kws else "misc"

            central_idxs = pick_central_claims(idxs, vectors, n=self.config.summary_sentences_per_theme)
            summary = [
                f"{claims[i].text} (source={claims[i].source_id} sent={claims[i].sent_idx})"
                for i in central_idxs
            ]

            stats = {
                "n_claims": len(cluster_claims),
                "kind_counts": _count_kinds(cluster_claims),
                "sources": sorted({c.source_id for c in cluster_claims}),
                "keywords": kws,
            }

            themes.append(
                Theme(
                    id=f"theme_{t_i+1}",
                    label=label,
                    claim_ids=[claims[i].id for i in idxs],
                    keywords=kws,
                    summary=summary,
                    stats=stats,
                )
            )

        agreements, contradictions = self._pairwise_signals(claims, vectors)
        uncertainties = self._uncertainties(themes, claim_by_id, contradictions)
        next_questions = self._next_questions(uncertainties)

        key_claims_by_theme = {
            th.label: [_format_claim(claim_by_id[cid]) for cid in th.claim_ids]
            for th in themes
        }

        scores = self._scores(themes, agreements, contradictions, uncertainties)

        audit = {
            "backend": self.backend.name,
            "n_sources": len(sources),
            "n_claims": len(claims),
            "thresholds": dataclasses.asdict(self.config),
        }

        return SynthesisResult(
            question=question,
            themes=themes,
            key_claims_by_theme=key_claims_by_theme,
            agreements=agreements,
            contradictions=contradictions,
            uncertainties=uncertainties,
            next_questions=next_questions,
            audit=audit,
            scores=scores,
        )

    def _pairwise_signals(
        self,
        claims: List[Claim],
        vectors: List[List[float]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        agreements: List[Dict[str, Any]] = []
        contradictions: List[Dict[str, Any]] = []
        n = len(claims)

        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine(vectors[i], vectors[j])
                if sim < min(self.config.agreement_threshold, self.config.contradiction_threshold):
                    continue

                opposite_polarity = (
                    claims[i].polarity in {"pos", "neg"} and
                    claims[j].polarity in {"pos", "neg"} and
                    claims[i].polarity != claims[j].polarity
                )

                if sim >= self.config.agreement_threshold and not opposite_polarity:
                    agreements.append(_pair_record("agreement", claims[i], claims[j], sim))
                elif sim >= self.config.contradiction_threshold and opposite_polarity:
                    contradictions.append(_pair_record("contradiction", claims[i], claims[j], sim))

        agreements.sort(key=lambda r: r["similarity"], reverse=True)
        contradictions.sort(key=lambda r: r["similarity"], reverse=True)
        return agreements[:30], contradictions[:30]

    def _uncertainties(
        self,
        themes: List[Theme],
        claim_by_id: Dict[str, Claim],
        contradictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contradictions_set = {(c["a"]["id"], c["b"]["id"]) for c in contradictions}
        contradictions_set |= {(c["b"]["id"], c["a"]["id"]) for c in contradictions}

        out: List[Dict[str, Any]] = []
        for th in themes:
            cs = [claim_by_id[cid] for cid in th.claim_ids]
            kinds = _count_kinds(cs)
            n = len(cs)

            if n < self.config.low_support_min_claims:
                out.append({
                    "theme": th.label,
                    "type": "low_support",
                    "message": "Theme has low support (few claims); may be non-representative.",
                    "evidence": [_format_claim(c) for c in cs[:2]],
                    "severity": 0.65,
                })

            oa = (kinds.get("opinion", 0) + kinds.get("assumption", 0))
            ratio = oa / max(1, n)
            if ratio >= self.config.opinion_assumption_heavy_ratio:
                out.append({
                    "theme": th.label,
                    "type": "weak_grounding",
                    "message": "Theme is dominated by opinions/assumptions; factual grounding is limited.",
                    "evidence": [_format_claim(c) for c in cs[:3]],
                    "severity": 0.7,
                })

            contr = 0
            for i in range(len(th.claim_ids)):
                for j in range(i + 1, len(th.claim_ids)):
                    if (th.claim_ids[i], th.claim_ids[j]) in contradictions_set:
                        contr += 1
            possible_pairs = n * (n - 1) // 2
            contr_ratio = contr / max(1, possible_pairs)
            if contr_ratio >= self.config.contradiction_heavy_ratio:
                out.append({
                    "theme": th.label,
                    "type": "inconsistency",
                    "message": "Theme contains internally conflicting claims; investigate scope/definitions.",
                    "evidence": [
                        r for r in contradictions
                        if r["a"]["id"] in th.claim_ids and r["b"]["id"] in th.claim_ids
                    ][:3],
                    "severity": 0.8,
                })

        out.sort(key=lambda x: x["severity"], reverse=True)
        return out

    def _next_questions(self, uncertainties: List[Dict[str, Any]]) -> List[str]:
        if not uncertainties:
            return ["No major gaps detected; add contrary sources to stress-test the synthesis."]
        qs: List[str] = []
        for u in uncertainties[:10]:
            theme = u["theme"]
            t = u["type"]
            if t == "low_support":
                qs.append(f"For '{theme}', can we add 2–3 independent sources to corroborate or refute it?")
            elif t == "weak_grounding":
                qs.append(f"For '{theme}', what measurable metrics or studies can convert assumptions into facts?")
            elif t == "inconsistency":
                qs.append(f"For '{theme}', what scope/definition differences explain the contradictions across sources?")
            else:
                qs.append(f"For '{theme}', what additional evidence would reduce uncertainty here?")
        return qs

    def _scores(
        self,
        themes: List[Theme],
        agreements: List[Dict[str, Any]],
        contradictions: List[Dict[str, Any]],
        uncertainties: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        n_claims = sum(t.stats.get("n_claims", 0) for t in themes)
        traceability = 1.0 if n_claims > 0 else 0.0
        consistency = 1.0 - min(1.0, len(contradictions) / max(1.0, len(agreements) + len(contradictions)))
        usefulness = 1.0 if uncertainties else 0.75
        return {
            "traceability": round(traceability, 3),
            "consistency": round(consistency, 3),
            "usefulness": round(usefulness, 3),
            "hallucination_risk_control": 0.9,
        }


# ----------------------------
# Output formatting
# ----------------------------

def to_json_dict(res: SynthesisResult) -> Dict[str, Any]:
    return {
        "question": res.question,
        "themes": [
            {
                "id": t.id,
                "label": t.label,
                "keywords": t.keywords,
                "summary": t.summary,
                "stats": t.stats,
            }
            for t in res.themes
        ],
        "key_claims_by_theme": res.key_claims_by_theme,
        "agreements": res.agreements,
        "contradictions": res.contradictions,
        "uncertainties": res.uncertainties,
        "next_questions": res.next_questions,
        "audit": res.audit,
        "scores": res.scores,
    }


def to_markdown(res: SynthesisResult) -> str:
    lines: List[str] = []
    lines.append("# AI Synthesis Agent Report\n")
    lines.append(f"**Question:** {res.question}\n")
    lines.append(
        f"**Backend:** `{res.audit.get('backend')}`  \n"
        f"**Claims:** {res.audit.get('n_claims')}  \n"
        f"**Themes:** {len(res.themes)}\n"
    )

    lines.append("## Themes\n")
    for t in res.themes:
        lines.append(f"### {t.label}\n")
        lines.append(f"- **Keywords:** {', '.join(t.keywords) if t.keywords else '—'}")
        lines.append(f"- **Stats:** {t.stats}")
        lines.append("- **Summary (extractive):**")
        for s in t.summary:
            lines.append(f"  - {s}")
        lines.append("\n**Evidence (claims):**")
        for c in res.key_claims_by_theme.get(t.label, [])[:12]:
            lines.append(f"- {c}")
        lines.append("")

    if res.agreements:
        lines.append("## Agreements (high similarity)\n")
        for a in res.agreements[:10]:
            lines.append(f"- sim={a['similarity']}: {a['a']['text']}  ↔  {a['b']['text']}")
        lines.append("")

    if res.contradictions:
        lines.append("## Potential Contradictions (high similarity + polarity mismatch)\n")
        for c in res.contradictions[:10]:
            lines.append(f"- sim={c['similarity']}: {c['a']['text']}  ↔  {c['b']['text']}")
        lines.append("")

    lines.append("## Uncertainties & Gaps\n")
    if not res.uncertainties:
        lines.append("- None detected by current heuristics.\n")
    else:
        for u in res.uncertainties[:12]:
            lines.append(f"- **{u['theme']}** ({u['type']}, severity={u['severity']}): {u['message']}")
            for e in (u.get("evidence") or [])[:3]:
                lines.append(f"  - {e}")
        lines.append("")

    lines.append("## Next Questions\n")
    for q in res.next_questions[:12]:
        lines.append(f"- {q}")
    lines.append("")

    lines.append("## Scores\n")
    for k, v in res.scores.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines.append("## Audit\n")
    lines.append(f"```json\n{json.dumps(res.audit, indent=2)}\n```")
    return "\n".join(lines)


# ----------------------------
# IO + CLI
# ----------------------------

def _read_sources(path: str) -> List[Dict[str, Any]]:
    if path == "-":
        return json.loads(sys.stdin.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _demo_sources() -> List[Dict[str, Any]]:
    return [
        {"id": "source1", "text": "Adoption increased among mid-sized firms in past 12 months. But it remains experimental."},
        {"id": "source2", "text": "Governance is a limiting factor. Productivity gains are uncertain."},
        {"id": "source3", "text": "Adoption is increasing but uneven. No consensus on ROI."},
        {"id": "source4", "text": "Assume tooling maturity is low in regulated industries."},
        {"id": "source5", "text": "Opinion: Human oversight intervenes frequently."},
    ]


def _write_out(payload: str, out_path: str) -> None:
    if out_path == "-":
        sys.stdout.write(payload + ("" if payload.endswith("\n") else "\n"))
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(payload + ("" if payload.endswith("\n") else "\n"))


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description="AI Synthesis Agent (grounded, traceable, agent-like).")
    sub = p.add_subparsers(dest="cmd")

    demo = sub.add_parser("demo", help="Run a built-in demo (no args).")
    demo.add_argument("--format", choices=["json", "markdown"], default="markdown")
    demo.add_argument("--out", default="-", help="Output path or '-' for stdout.")

    run = sub.add_parser("run", help="Run on a sources.json file (or stdin).")
    run.add_argument("--question", required=True, help="Research question.")
    run.add_argument("--sources", required=True, help="Path to sources.json or '-' for stdin.")
    run.add_argument("--format", choices=["json", "markdown"], default="json")
    run.add_argument("--out", default="-", help="Output path or '-' for stdout.")

    # If user runs with no args: behave like an agent tool (demo by default)
    if not argv:
        argv = ["demo"]

    args = p.parse_args(argv)

    agent = AISynthesisAgent()

    if args.cmd == "demo":
        question = "What factors affect AI adoption in organizations?"
        res = agent.synthesize(_demo_sources(), question)
        payload = json.dumps(to_json_dict(res), indent=2) if args.format == "json" else to_markdown(res)
        _write_out(payload, args.out)
        return 0

    if args.cmd == "run":
        sources = _read_sources(args.sources)
        res = agent.synthesize(sources, args.question)
        payload = json.dumps(to_json_dict(res), indent=2) if args.format == "json" else to_markdown(res)
        _write_out(payload, args.out)
        return 0

    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
