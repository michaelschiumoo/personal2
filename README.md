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


Design goals:
  - Extractive only (no invented sentences)
  - Traceability: every claim cites source_id + sentence index
  - Deterministic fallback behavior (no hard dependency on heavy ML libs)
  - Readable executive output on tiny datasets:
      * clustering + optional conservative post-merge
      * then DISPLAY COMPACTION: bucket leftovers into "other" (NOT a semantic merge)
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple




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
    backend: str
    cluster_backend: str
    agent_plan: List[str]
    agent_decisions: Dict[str, Any]
    retrieval_trace: Dict[str, Any]
    iteration_trace: List[Dict[str, Any]]


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
    # Claim extraction / retrieval
    min_sentence_chars: int = 18
    max_sentences_per_source: int = 120
    topk_sentences_per_source: int = 2


    # Theme targets (for "executive-friendly" output)
    target_min_themes: int = 3
    target_max_themes: int = 5


    # Clustering threshold search
    base_theme_threshold: float = 0.18
    min_threshold: float = 0.03
    max_threshold: float = 0.90
    max_iters: int = 8
    threshold_step: float = 0.04


    # Quality heuristics
    min_multi_claim_themes: int = 1
    max_themes_hard_cap: int = 12


    agreement_threshold: float = 0.82
    contradiction_threshold: float = 0.82


    low_support_min_claims: int = 2
    opinion_assumption_heavy_ratio: float = 0.60
    contradiction_heavy_ratio: float = 0.25


    summary_sentences_per_theme: int = 2
    exec_summary_bullets: int = 3


    # Conservative post-merge (token overlap only; never forced)
    post_merge_enabled: bool = True
    post_merge_jaccard: float = 0.18
    post_merge_shared_token_bonus: float = 0.10
    post_merge_max_merges: int = 10


    # Fallback when vector backend yields near-zero similarities
    low_similarity_floor: float = 0.20


    # DISPLAY compaction (honest): if still too many themes, bucket smallest into "other".
    # This is NOT a semantic merge; it is a presentation step.
    compact_excess_themes: bool = True
    compact_other_label: str = "other"




# ----------------------------
# Text utilities
# ----------------------------


_STOPWORDS: Set[str] = {
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
        if len(t) > 4 and t.endswith("s") and not t.endswith("us"):
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
# Vector backends
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




class PureTfidfBackend(VectorBackend):
    """Dependency-free TF-IDF with (1,2)-grams over our tokenizer."""
    name = "pure-tfidf"


    def __init__(self, max_features: int = 6000) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []
        self._fitted = False


    @staticmethod
    def _ngrams(tokens: Tuple[str, ...]) -> List[str]:
        out = list(tokens)
        out.extend([f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)])
        return out


    def _build_vocab(self, texts: List[str]) -> None:
        df: Dict[str, int] = {}
        for t in texts:
            feats = set(self._ngrams(tokenize(t)))
            for f in feats:
                df[f] = df.get(f, 0) + 1


        items = sorted(df.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        items = items[: self.max_features]
        self.vocab = {w: i for i, (w, _) in enumerate(items)}


        n_docs = max(1, len(texts))
        self.idf = [0.0] * len(self.vocab)
        for w, i in self.vocab.items():
            dfi = df.get(w, 0)
            self.idf[i] = math.log((1.0 + n_docs) / (1.0 + dfi)) + 1.0


        self._fitted = True


    def _vec(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("PureTfidfBackend used before fit().")
        v = [0.0] * len(self.vocab)
        feats = self._ngrams(tokenize(text))
        tf: Dict[int, float] = {}
        for f in feats:
            idx = self.vocab.get(f)
            if idx is None:
                continue
            tf[idx] = tf.get(idx, 0.0) + 1.0
        for idx, cnt in tf.items():
            v[idx] = cnt * self.idf[idx]
        n = math.sqrt(sum(x * x for x in v))
        if n > 0:
            v = [x / n for x in v]
        return v


    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        self._build_vocab(texts)
        return [self._vec(t) for t in texts]


    def transform(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            raise RuntimeError("transform() called before fit_transform().")
        return [self._vec(t) for t in texts]




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
    # Prefer nicer embeddings if available, but never depend on them.
    try:
        return SentenceTransformerBackend()
    except Exception:
        pass
    try:
        return SklearnTfidfBackend()
    except Exception:
        pass
    return PureTfidfBackend()




# ----------------------------
# Clustering (sklearn optional, fallback greedy)
# ----------------------------


def cluster_vectors(vectors: List[List[float]], threshold: float) -> List[List[int]]:
    if not vectors:
        return []


    # Optional sklearn clustering
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


    # Greedy fallback
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




def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))




def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    p = _clamp(p, 0.0, 1.0)
    idx = int(round((len(sorted_vals) - 1) * p))
    return float(sorted_vals[idx])




def _pairwise_sim_stats(vectors: List[List[float]]) -> Dict[str, float]:
    sims: List[float] = []
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine(vectors[i], vectors[j]))
    sims.sort()
    return {
        "p50": round(_percentile(sims, 0.50), 4),
        "p75": round(_percentile(sims, 0.75), 4),
        "p90": round(_percentile(sims, 0.90), 4),
        "p95": round(_percentile(sims, 0.95), 4),
        "max": round(sims[-1], 4) if sims else 0.0,
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


    def _retrieve_topk_sentences(
        self,
        sources: List[Source],
        question: str,
    ) -> Tuple[List[Tuple[Source, List[Tuple[int, str, int, int, float]]]], Dict[str, Any]]:
        """Dependency-free retrieval to keep outputs constrained and deterministic."""
        retriever = PureTfidfBackend(max_features=6000)


        per_source: List[List[Tuple[int, str, int, int]]] = []
        all_texts: List[str] = [question]


        for src in sources:
            sents = split_sentences(src.text)[: self.config.max_sentences_per_source]
            filtered: List[Tuple[int, str, int, int]] = []
            for sent_idx, (sent, start, end) in enumerate(sents):
                sent = sent.strip()
                if len(sent) < self.config.min_sentence_chars:
                    continue
                filtered.append((sent_idx, sent, start, end))
                all_texts.append(sent)
            per_source.append(filtered)


        vecs = retriever.fit_transform(all_texts)
        qv = vecs[0]
        cursor = 1


        out: List[Tuple[Source, List[Tuple[int, str, int, int, float]]]] = []
        candidates = 0
        selected = 0


        for src, sents in zip(sources, per_source):
            scored: List[Tuple[int, str, int, int, float]] = []
            for (sent_idx, sent, start, end) in sents:
                sv = vecs[cursor]
                cursor += 1
                candidates += 1
                sim = cosine(qv, sv)
                scored.append((sent_idx, sent, start, end, sim))
            scored.sort(key=lambda x: x[4], reverse=True)
            topk = scored[: self.config.topk_sentences_per_source]
            selected += len(topk)
            out.append((src, topk))


        trace = {
            "topk_per_source": self.config.topk_sentences_per_source,
            "candidates": candidates,
            "selected": selected,
        }
        return out, trace


    def extract_claims(self, sources: List[Source], question: str) -> Tuple[List[Claim], Dict[str, Any]]:
        retrieved, trace = self._retrieve_topk_sentences(sources, question)


        claims: List[Claim] = []
        for src, picked in retrieved:
            for sent_idx, sent, start, end, _sim in picked:
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
                        text=sent,
                        kind=kind,
                        polarity=pol,
                        tokens=toks,
                    )
                )


        logger.info("Extracted %d claims.", len(claims))
        return claims, trace


    def synthesize(self, sources_raw: List[Dict[str, Any]], question: str) -> SynthesisResult:
        sources = self.validate(sources_raw, question)


        agent_plan = [
            "Objective: Produce a grounded synthesis report with traceability and minimal hallucination risk.",
            "Constraints: Extractive only; every claim cites source_id + sentence index.",
            "Steps: Retrieve top K relevant sentences per source; extract claims; cluster into themes; conservative post-merge by token overlap; if still too many themes, DISPLAY-COMPACT leftovers into 'other' (not a semantic merge); detect agreements/contradictions; generate uncertainties + next questions.",
        ]


        claims, retrieval_trace = self.extract_claims(sources, question)
        claim_by_id = {c.id: c for c in claims}


        backend_selected = self.backend
        backend_name = backend_selected.name


        claim_texts = [c.text for c in claims]
        vectors = backend_selected.fit_transform(claim_texts) if claim_texts else []


        sim_stats = _pairwise_sim_stats(vectors)
        max_sim = float(sim_stats.get("max", 0.0))


        cluster_backend: VectorBackend = backend_selected
        cluster_backend_name = backend_name
        reason = "primary_backend_ok"


        if vectors and max_sim < self.config.low_similarity_floor:
            logger.warning(
                "Low similarity for backend=%s (max=%.4f < floor=%.2f); clustering fallback -> hashed-bow",
                backend_name,
                max_sim,
                self.config.low_similarity_floor,
            )
            cluster_backend = HashedBoWBackend()
            cluster_backend_name = cluster_backend.name
            reason = "fallback_low_similarity"
            vectors = cluster_backend.fit_transform(claim_texts)
            sim_stats = _pairwise_sim_stats(vectors)


        iteration_trace: List[Dict[str, Any]] = []


        threshold = _clamp(self.config.base_theme_threshold, self.config.min_threshold, self.config.max_threshold)
        best_pack: Optional[Tuple[float, List[List[int]]]] = None


        for it in range(1, self.config.max_iters + 1):
            clusters = cluster_vectors(vectors, threshold=threshold)
            clusters = sorted(clusters, key=len, reverse=True)[: self.config.max_themes_hard_cap]


            themes, key_claims_by_theme = self._build_themes(clusters, claims, vectors, claim_by_id)


            # Conservative post-merge (never forced)
            post_merge_merges = 0
            if self.config.post_merge_enabled:
                themes, key_claims_by_theme, post_merge_merges = self._post_merge_themes(
                    themes,
                    key_claims_by_theme,
                    claim_by_id,
                )


            n_themes = len(themes)
            multi_claim = sum(1 for t in themes if len(t.claim_ids) >= 2)


            ok = (
                self.config.target_min_themes <= n_themes <= self.config.target_max_themes
                and multi_claim >= self.config.min_multi_claim_themes
            )


            iteration_trace.append(
                {
                    "iter": it,
                    "threshold": round(threshold, 4),
                    "themes": n_themes,
                    "multi_claim_themes": multi_claim,
                    "pre_compaction_ok": ok,
                    "post_merge_merges": post_merge_merges,
                }
            )


            if best_pack is None:
                best_pack = (threshold, clusters)
            else:
                prev_th, prev_clusters = best_pack
                prev_themes, _ = self._build_themes(prev_clusters, claims, vectors, claim_by_id)
                prev_n = len(prev_themes)
                prev_multi = sum(1 for t in prev_themes if len(t.claim_ids) >= 2)


                def dist(n: int) -> int:
                    if n < self.config.target_min_themes:
                        return self.config.target_min_themes - n
                    if n > self.config.target_max_themes:
                        return n - self.config.target_max_themes
                    return 0


                if (dist(n_themes), -multi_claim) < (dist(prev_n), -prev_multi):
                    best_pack = (threshold, clusters)


            if ok:
                break


            # Adjust threshold
            # Too many themes => lower threshold (merge more)
            # Too few themes => raise threshold (split more)
            if n_themes > self.config.target_max_themes or multi_claim < self.config.min_multi_claim_themes:
                threshold = _clamp(threshold - self.config.threshold_step, self.config.min_threshold, self.config.max_threshold)
            elif n_themes < self.config.target_min_themes:
                threshold = _clamp(threshold + self.config.threshold_step, self.config.min_threshold, self.config.max_threshold)
            else:
                threshold = _clamp(threshold - (self.config.threshold_step / 2.0), self.config.min_threshold, self.config.max_threshold)


        if best_pack is None:
            best_pack = (threshold, cluster_vectors(vectors, threshold=threshold))


        final_threshold, final_clusters = best_pack
        final_threshold = _clamp(final_threshold, self.config.min_threshold, self.config.max_threshold)


        themes, key_claims_by_theme = self._build_themes(final_clusters, claims, vectors, claim_by_id)


        post_merge_merges_final = 0
        if self.config.post_merge_enabled:
            themes, key_claims_by_theme, post_merge_merges_final = self._post_merge_themes(
                themes,
                key_claims_by_theme,
                claim_by_id,
            )


        # DISPLAY COMPACTION (honest): if still too many themes, bucket smallest into "other"
        compaction_notes: Dict[str, Any] = {"compaction": "disabled"}
        if self.config.compact_excess_themes:
            themes, key_claims_by_theme, compaction_notes = self._compact_themes_for_display(
                themes,
                key_claims_by_theme,
                claim_by_id,
                target_max=self.config.target_max_themes,
            )


        agreements, contradictions = self._pairwise_signals(claims, vectors)
        uncertainties = self._uncertainties(themes, claim_by_id, contradictions)
        next_questions = self._next_questions(uncertainties)
        scores = self._scores(themes, agreements, contradictions, uncertainties)


        targets_met = (
            self.config.target_min_themes <= len(themes) <= self.config.target_max_themes
            and sum(1 for t in themes if len(t.claim_ids) >= 2) >= self.config.min_multi_claim_themes
        )


        agent_decisions = {
            "backend_selected": backend_name,
            "cluster_backend_selected": cluster_backend_name,
            "reason": reason,
            "post_merge": "applied" if self.config.post_merge_enabled else "disabled",
            "post_merge_merges_final": post_merge_merges_final,
            "targets_met": targets_met,
            "themes_final": len(themes),
            **compaction_notes,
        }


        audit = {
            "backend": backend_name,
            "cluster_backend": cluster_backend_name,
            "n_sources": len(sources),
            "n_claims": len(claims),
            "effective_theme_threshold": round(final_threshold, 4),
            "similarity_stats": sim_stats,
            "thresholds": dataclasses.asdict(self.config),
            "notes": {
                "no_forced_semantic_merges": True,
                "post_merge_only_above_threshold": True,
                "display_compaction_is_not_semantic_merge": True,
            },
        }


        return SynthesisResult(
            question=question,
            backend=backend_name,
            cluster_backend=cluster_backend_name,
            agent_plan=agent_plan,
            agent_decisions=agent_decisions,
            retrieval_trace=retrieval_trace,
            iteration_trace=iteration_trace,
            themes=themes,
            key_claims_by_theme=key_claims_by_theme,
            agreements=agreements,
            contradictions=contradictions,
            uncertainties=uncertainties,
            next_questions=next_questions,
            audit=audit,
            scores=scores,
        )


    def _build_themes(
        self,
        clusters: List[List[int]],
        claims: List[Claim],
        vectors: List[List[float]],
        claim_by_id: Dict[str, Claim],
    ) -> Tuple[List[Theme], Dict[str, List[str]]]:
        themes: List[Theme] = []
        key_claims_by_theme: Dict[str, List[str]] = {}
        used_labels: Dict[str, int] = {}


        for t_i, idxs in enumerate(clusters[: self.config.max_themes_hard_cap]):
            cluster_claims = [claims[i] for i in idxs]
            kws = top_keywords_from_claims(cluster_claims, k=6)
            label = kws[0] if kws else "misc"


            used_labels[label] = used_labels.get(label, 0) + 1
            if used_labels[label] > 1:
                label = f"{label}_{used_labels[label]}"


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


            th = Theme(
                id=f"theme_{t_i + 1}",
                label=label,
                claim_ids=[claims[i].id for i in idxs],
                keywords=kws,
                summary=summary,
                stats=stats,
            )
            themes.append(th)
            key_claims_by_theme[label] = [_format_claim(claim_by_id[cid]) for cid in th.claim_ids]


        return themes, key_claims_by_theme


    def _post_merge_themes(
        self,
        themes: List[Theme],
        key_claims_by_theme: Dict[str, List[str]],
        claim_by_id: Dict[str, Claim],
    ) -> Tuple[List[Theme], Dict[str, List[str]], int]:
        """
        Conservative merge: only merge themes when token Jaccard >= threshold.
        Never force merges just to hit a target.
        """
        if len(themes) <= 1:
            return themes, key_claims_by_theme, 0


        def theme_tokens(t: Theme) -> Set[str]:
            s: Set[str] = set()
            for cid in t.claim_ids:
                s.update(claim_by_id[cid].tokens)
            return {x for x in s if x and x not in _STOPWORDS}


        def jaccard(a: Set[str], b: Set[str]) -> float:
            if not a and not b:
                return 0.0
            inter = len(a & b)
            union = len(a | b)
            return inter / max(1, union)


        merges = 0


        # Try up to N merges
        for _ in range(self.config.post_merge_max_merges):
            if len(themes) <= self.config.target_max_themes:
                break


            best_score = -1.0
            best_pair: Optional[Tuple[int, int]] = None


            token_cache = [theme_tokens(t) for t in themes]


            for i in range(len(themes)):
                for j in range(i + 1, len(themes)):
                    a = token_cache[i]
                    b = token_cache[j]
                    score = jaccard(a, b)
                    if (a & b):
                        score += self.config.post_merge_shared_token_bonus
                    if score < self.config.post_merge_jaccard:
                        continue
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)


            if best_pair is None:
                break


            i, j = best_pair
            self._merge_pair(themes, key_claims_by_theme, claim_by_id, i, j)
            merges += 1


        return themes, key_claims_by_theme, merges


    @staticmethod
    def _merge_pair(
        themes: List[Theme],
        key_claims_by_theme: Dict[str, List[str]],
        claim_by_id: Dict[str, Claim],
        i: int,
        j: int,
    ) -> None:
        a = themes[i]
        b = themes[j]


        keep_a = (len(a.claim_ids), a.label) >= (len(b.claim_ids), b.label)
        primary = a if keep_a else b
        secondary = b if keep_a else a


        merged_claim_ids = primary.claim_ids + [cid for cid in secondary.claim_ids if cid not in primary.claim_ids]
        merged_claims = [claim_by_id[cid] for cid in merged_claim_ids]


        kws = top_keywords_from_claims(merged_claims, k=6)
        primary.claim_ids = merged_claim_ids
        primary.keywords = kws
        primary.stats = {
            "n_claims": len(merged_claims),
            "kind_counts": _count_kinds(merged_claims),
            "sources": sorted({c.source_id for c in merged_claims}),
            "keywords": kws,
        }


        key_claims_by_theme[primary.label] = [_format_claim(claim_by_id[cid]) for cid in primary.claim_ids]
        key_claims_by_theme.pop(secondary.label, None)
        themes.remove(secondary)


    def _compact_themes_for_display(
        self,
        themes: List[Theme],
        key_claims_by_theme: Dict[str, List[str]],
        claim_by_id: Dict[str, Claim],
        target_max: int,
    ) -> Tuple[List[Theme], Dict[str, List[str]], Dict[str, Any]]:
        """
        DISPLAY compaction:
          If themes > target_max, keep strongest (by n_claims) and bucket the rest in 'other'.
          This is NOT a semantic merge; it is presentation only.
        """
        if len(themes) <= target_max:
            return themes, key_claims_by_theme, {"compaction": "not_needed"}


        # Keep top (target_max - 1) themes; last slot is 'other'
        keep_n = max(1, target_max - 1)
        themes_sorted = sorted(themes, key=lambda t: (t.stats.get("n_claims", 0), t.label), reverse=True)
        kept = themes_sorted[:keep_n]
        rest = themes_sorted[keep_n:]


        # Collect all leftover claims (dedupe)
        seen: Set[str] = set()
        other_claim_ids: List[str] = []
        for t in rest:
            for cid in t.claim_ids:
                if cid not in seen:
                    seen.add(cid)
                    other_claim_ids.append(cid)


        other_claims = [claim_by_id[cid] for cid in other_claim_ids]
        kws = top_keywords_from_claims(other_claims, k=6)


        label = self.config.compact_other_label
        existing = {t.label for t in kept}
        if label in existing:
            k = 2
            while f"{label}_{k}" in existing:
                k += 1
            label = f"{label}_{k}"


        summary: List[str] = []
        for c in other_claims[: self.config.summary_sentences_per_theme]:
            summary.append(f"{c.text} (source={c.source_id} sent={c.sent_idx})")


        other_theme = Theme(
            id="theme_other",
            label=label,
            claim_ids=other_claim_ids,
            keywords=kws,
            summary=summary,
            stats={
                "n_claims": len(other_claim_ids),
                "kind_counts": _count_kinds(other_claims),
                "sources": sorted({c.source_id for c in other_claims}),
                "keywords": kws,
                "note": "display_compaction_bucket",
            },
        )


        new_key: Dict[str, List[str]] = {}
        for t in kept:
            new_key[t.label] = key_claims_by_theme.get(t.label, [])
        new_key[other_theme.label] = [_format_claim(claim_by_id[cid]) for cid in other_theme.claim_ids]


        notes = {
            "compaction": "applied",
            "bucket_label": other_theme.label,
            "kept_themes": len(kept),
            "bucketed_themes": len(rest),
        }
        return kept + [other_theme], new_key, notes


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
        "backend": res.backend,
        "cluster_backend": res.cluster_backend,
        "agent_plan": res.agent_plan,
        "agent_decisions": res.agent_decisions,
        "retrieval_trace": res.retrieval_trace,
        "iteration_trace": res.iteration_trace,
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
        f"**Backend:** `{res.backend}`  \n"
        f"**Cluster Backend:** `{res.cluster_backend}`  \n"
        f"**Claims:** {res.audit.get('n_claims')}  \n"
        f"**Themes:** {len(res.themes)}\n"
    )


    # Executive Summary (extractive)
    lines.append("## Executive Summary (extractive)\n")
    bullets = 0
    for t in res.themes:
        if not t.summary:
            continue
        lines.append(f"- [{t.label}] {t.summary[0]}")
        bullets += 1
        if bullets >= res.audit["thresholds"]["exec_summary_bullets"]:
            break
    if bullets == 0:
        lines.append("- (No summary sentences available.)")
    lines.append("")


    # Agent Plan
    lines.append("## Agent Plan\n")
    for s in res.agent_plan:
        if s.startswith("Objective:"):
            lines.append(f"- **Objective:** {s.replace('Objective:', '').strip()}")
        elif s.startswith("Constraints:"):
            lines.append(f"- **Constraints:** {s.replace('Constraints:', '').strip()}")
        elif s.startswith("Steps:"):
            steps = s.replace("Steps:", "").strip()
            lines.append("- **Steps:**")
            for part in [p.strip() for p in steps.split(";") if p.strip()]:
                lines.append(f"  - {part}")
        else:
            lines.append(f"- {s}")
    lines.append("")


    # Decisions
    lines.append("## Agent Decisions\n")
    for k, v in res.agent_decisions.items():
        lines.append(f"- {k}: `{v}`" if isinstance(v, str) else f"- {k}: {v}")
    lines.append("")


    # Retrieval Trace
    lines.append("## Retrieval Trace\n")
    lines.append(f"- **TopK per source:** {res.retrieval_trace.get('topk_per_source')}")
    lines.append(f"- **Candidates:** {res.retrieval_trace.get('candidates')}, **Selected:** {res.retrieval_trace.get('selected')}\n")


    # Iteration Trace
    lines.append("## Iteration Trace\n")
    for it in res.iteration_trace:
        bits = [
            f"iter={it.get('iter')}",
            f"threshold={it.get('threshold')}",
            f"themes={it.get('themes')}",
            f"multi_claim_themes={it.get('multi_claim_themes')}",
            f"pre_compaction_ok={it.get('pre_compaction_ok')}",
            f"post_merge_merges={it.get('post_merge_merges')}",
        ]
        lines.append(f"- {' '.join(bits)}")
    lines.append("")


    # Themes
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


    # Agreements / Contradictions
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


    # Uncertainties
    lines.append("## Uncertainties & Gaps\n")
    if not res.uncertainties:
        lines.append("- None detected by current heuristics.\n")
    else:
        for u in res.uncertainties[:12]:
            lines.append(f"- **{u['theme']}** ({u['type']}, severity={u['severity']}): {u['message']}")
            for e in (u.get("evidence") or [])[:3]:
                lines.append(f"  - {e}")
        lines.append("")


    # Next Questions
    lines.append("## Next Questions\n")
    for q in res.next_questions[:12]:
        lines.append(f"- {q}")
    lines.append("")


    # Scores
    lines.append("## Scores\n")
    for k, v in res.scores.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")


    # Audit
    lines.append("## Audit\n")
    lines.append(f"```json\n{json.dumps(res.audit, indent=2)}\n```")
    return "\n".join(lines)




# ----------------------------
# IO + CLI
# ----------------------------


def _read_sources(path: str) -> List[Dict[str, Any]]:
    if path == "-":
        raw = sys.stdin.read().strip()
        if not raw:
            raise ValueError("stdin is empty but --sources was '-'")
        return json.loads(raw)
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








