from collections import defaultdict

import numpy as np

_embedder = None
_reranker = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-m3")
        _embedder.max_seq_length = 1024
    return _embedder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


class Retriever:
    def __init__(self, collection: str = "docs_1200"):
        import chromadb
        self.col = chromadb.PersistentClient(path="chroma").get_collection(collection)
        self._bm25 = None

    def vector(self, q: str, k: int = 5) -> list[dict]:
        qv = get_embedder().encode([q], normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=qv, n_results=k)
        return [
            {"id": i, "text": d, "score": 1 - dist, **m}   # 코사인 거리 → 유사도
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0],
                                     r["metadatas"][0], r["distances"][0])
        ]

    def _tok(self, text: str) -> list[str]:
        return [t.form.lower() for t in self._kiwi.tokenize(text)]

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        from kiwipiepy import Kiwi
        from rank_bm25 import BM25Okapi
        self._kiwi = Kiwi()
        data = self.col.get(include=["documents", "metadatas"])
        self._ids, self._docs, self._metas = data["ids"], data["documents"], data["metadatas"]
        self._bm25 = BM25Okapi([self._tok(d) for d in self._docs])

    def keyword(self, q: str, k: int = 5) -> list[dict]:
        self._ensure_bm25()
        scores = self._bm25.get_scores(self._tok(q))
        top = np.argsort(scores)[::-1][:k]
        return [{"id": self._ids[i], "text": self._docs[i], "score": float(scores[i]), **self._metas[i]}
                for i in top]

    def hybrid(self, q: str, k: int = 5, pool: int = 20) -> list[dict]:
        fused, by_id = defaultdict(float), {}
        for results in (self.vector(q, pool), self.keyword(q, pool)):
            for rank, d in enumerate(results, start=1):
                fused[d["id"]] += 1 / (60 + rank)          # RRF
                by_id.setdefault(d["id"], d)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**by_id[i], "score": fused[i]} for i in best]

    def rerank(self, q: str, k: int = 5, pool: int = 20) -> list[dict]:
        cands = self.hybrid(q, pool, pool)
        scores = get_reranker().predict([(q, d["text"]) for d in cands])
        order = np.argsort(scores)[::-1][:k]
        return [{**cands[i], "score": float(scores[i])} for i in order]
    
    
    
    