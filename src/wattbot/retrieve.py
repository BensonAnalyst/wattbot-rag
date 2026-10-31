"""Hybrid retrieval: BM25 keywords + optional dense embeddings, fused with reciprocal rank fusion.

Questions often name the paper ("Dodge et al. (2022)", "Cottier et al.") or the model it studies. When the
question names an author of one of the 32 documents, retrieval is restricted to that author's documents.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this to was were what
which with when where who whom why will would can could does did do than then there these those per into over
about after before between according reported estimated using used""".split())


def tokenize(text: str) -> list:
    """Keep compound names whole ("gshard-600b", "ml.energy") and also add their parts ("gshard", "600b")."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()):
        parts = re.split(r"[.\-]", tok)
        out += [t for t in ([tok] + parts if len(parts) > 1 else [tok]) if t not in STOP]
    return out


class BM25:
    def __init__(self, docs: list, k1: float = 1.4, b: float = 0.75):
        self.tokens = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df = Counter(t for toks in self.tokens for t in set(toks))
        n = len(self.tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(toks) for toks in self.tokens]

    def scores(self, query: str) -> list:
        q = tokenize(query)
        out = []
        for tf, toks in zip(self.tf, self.tokens):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out.append(sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf))
        return out


def author_hints(question: str, authors: dict) -> set:
    """authors: {doc_id: first-author surname}. Returns the doc IDs whose first author the question names."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    named = {doc for doc, name in authors.items() if name and len(name) > 2 and name in words
             and re.search(rf"\b{name}\b\s+(et al|\(|and|\d{{4}})", question.lower())}
    return named


def rrf(rankings: list, k: int = 60) -> dict:
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self, chunks: list, authors: dict, embed_fn=None):
        """embed_fn(list[str]) -> list of unit-normalised vectors, or None for BM25 only."""
        self.chunks, self.authors, self.embed_fn = chunks, authors, embed_fn
        self.bm25 = BM25([c["search_text"] for c in chunks])
        self.vectors = np.asarray(embed_fn([c["search_text"] for c in chunks])) if embed_fn else None

    def search(self, question: str, k: int = 8, pool: int = 50) -> list:
        allowed = author_hints(question, self.authors)
        idx = [i for i, c in enumerate(self.chunks) if not allowed or c["doc_id"] in allowed]
        bm = self.bm25.scores(question)
        rankings = [[i for i in sorted(idx, key=lambda i: -bm[i])[:pool] if bm[i] > 0]]
        if self.vectors is not None:
            q = np.asarray(self.embed_fn([question])[0])
            sims = self.vectors[idx] @ q
            rankings.append([idx[j] for j in np.argsort(-sims)[:pool]])
        fused = rrf(rankings)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [{**self.chunks[i], "score": round(fused[i], 5)} for i in best]


def doc_recall(retrieved: list, truth_refs: set) -> float:
    """Share of the ground-truth documents that appear in the retrieved chunks."""
    if not truth_refs or truth_refs == {"is_blank"}:
        return None
    got = {c["doc_id"].lower() for c in retrieved}
    return len(got & truth_refs) / len(truth_refs)
