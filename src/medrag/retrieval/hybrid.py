"""Hybrid Retriever — Dense + BM25 결과를 RRF로 통합"""

from medrag.config import settings
from medrag.ingestion.embedder import Embedder
from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.retriever import DenseRetriever, RetrievedChunk


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion (RRF) 알고리즘.

    여러 검색 결과 리스트를 순위 기반으로 통합한다.
    RRF score = Σ 1 / (k + rank_i)

    k=60은 Cormack et al.(2009) 논문에서 제안한 기본값.
    낮은 k는 상위 랭킹에 더 집중하고, 높은 k는 전반적으로 고르게 취급한다.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # chunk_id를 키로 사용 (없으면 내용 해시로 대체)
            chunk_id = chunk.metadata.get("chunk_id", chunk.content[:64])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    results = []
    for cid in sorted_ids:
        chunk = chunk_map[cid]
        # RRF 점수를 메타데이터에도 기록 (디버깅용)
        fused_chunk = RetrievedChunk(
            content=chunk.content,
            score=round(scores[cid], 6),
            metadata={**chunk.metadata, "rrf_score": round(scores[cid], 6)},
        )
        results.append(fused_chunk)
    return results


class HybridRetriever:
    """
    Dense(벡터) + BM25(키워드) 검색 결과를 RRF로 통합하는 Hybrid Retriever.

    - Dense: 의미적 유사도 기반, 표현이 다른 동의어에 강함
    - BM25: 정확한 키워드 매칭, 의료 코드/수치 검색에 강함
    - RRF: 두 결과를 순위 기반으로 통합 → 단일 방식 대비 일관된 성능 향상
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self._dense = DenseRetriever(embedder=embedder or Embedder())
        self._bm25 = bm25_index or BM25Index()

    @property
    def bm25(self) -> BM25Index:
        return self._bm25

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or settings.retriever_top_k

        dense_results = self._dense.retrieve(query, top_k=k, score_threshold=0.0)
        bm25_results = self._bm25.search(query, top_k=k)

        if not dense_results and not bm25_results:
            return []

        # 두 결과 모두 있을 때 RRF, 하나만 있을 때 해당 결과 반환
        if dense_results and bm25_results:
            fused = _reciprocal_rank_fusion([dense_results, bm25_results])
        elif dense_results:
            fused = dense_results
        else:
            fused = bm25_results

        # score_threshold 적용 (RRF 점수 기준이 아닌 원본 dense score 기준)
        # RRF 점수는 절대값 의미가 없으므로 상위 top_k만 반환
        return fused[:k]
