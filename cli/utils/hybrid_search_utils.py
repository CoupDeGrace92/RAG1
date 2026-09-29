import os

from utils.search_utils import InvertedIndex, normalize_score_dicts, SearchResult, DocumentObject, rrf
from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        self.idx.load_or_build()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm_results = self._bm25_search(query, 500*limit)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        #Normalizations
        bm_results = normalize_score_dicts(bm_results)
        semantic_results = normalize_score_dicts(semantic_results)
        combined_map = {}
        for item in bm_results:
            combined_map[item.get("id")] = {
                "id": item.get("id"),
                "title": item.get("title"),
                "document": item.get("document"),
                "keyword_score": item.get("score")
            }
        for item in semantic_results:
            if combined_map.get(item.get("id")) is None:
                combined_map[item.get("id")] = {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "document": item.get("document"),
                    "semantic_score": item.get("score")
                }
            else:
                combined_map[item.get("id")]["semantic_score"] = item.get("score")

        for item in combined_map:
            d = combined_map[item]
            s = d.get("semantic_score") or 0
            k = d.get("keyword_score") or 0
            weighted = (k * alpha) + (s * (1-alpha))
            d["weighted"] = weighted

        return [item[1] for item in sorted(combined_map.items(), key=lambda item: item[1]["weighted"], reverse=True)][:limit]

    def rrf_search(self, query: str, k: int, limit: int=5) -> list[dict]:
        bm_results = self._bm25_search(query, 500*limit)
        semantic_results = self.semantic_search.search_chunks(query, 500*limit)

        #Normalizations
        bm_results = normalize_score_dicts(bm_results)
        semantic_results = normalize_score_dicts(semantic_results)
        combined_map = {}

        for idx, item in enumerate(bm_results):
            combined_map[item.get("id")] = {
                "id": item.get("id"),
                "title": item.get("title"),
                "document": item.get("document"),
                "bm_rank": idx+1
            }
        for idx, item in enumerate(semantic_results):
            if combined_map.get(item.get("id")) is None:
                combined_map[item.get("id")] = {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "document": item.get("document"),
                    "sem_rank": idx+1
                }
            else:
                combined_map[item.get("id")]["sem_rank"] = idx+1

        for item in combined_map:
            d = combined_map[item]
            b = d.get("bm_rank") or 0
            s = d.get("sem_rank") or 0
            r = rrf(b, k) + rrf(s, k)
            d["rrf"] = r

        return[item[1] for item in sorted(combined_map.items(), key=lambda item: item[1]["rrf"], reverse=True)][:limit]


