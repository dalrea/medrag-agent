"""Cross-encoder Reranker — 검색 결과 관련도 재정렬"""

from sentence_transformers import CrossEncoder

from medrag.config import settings
from medrag.retrieval.retriever import RetrievedChunk

# 의료/일반 도메인 재정렬에 검증된 Cross-encoder 모델
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder 기반 Reranker.

    Bi-encoder(Dense Retriever)는 질의와 문서를 독립적으로 임베딩하여 속도가 빠르지만
    질의-문서 간 세밀한 상호작용을 포착하지 못한다.
    Cross-encoder는 질의와 문서를 함께 입력받아 관련도를 직접 계산하므로 정밀도가 높다.

    단, 속도가 느리므로 Retriever가 좁힌 상위 N개에만 적용(Retrieve & Rerank 패턴).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model = CrossEncoder(model_name)
        self._top_k = settings.reranker_top_k

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        검색된 청크를 cross-encoder로 재정렬하여 상위 top_k 반환.

        Args:
            query: 사용자 질의
            chunks: Retriever가 반환한 후보 청크
            top_k: 반환할 최대 청크 수 (None이면 settings 값 사용)

        Returns:
            관련도 내림차순으로 정렬된 청크 리스트
        """
        k = top_k or self._top_k
        if not chunks:
            return []

        # Cross-encoder는 (query, document) 쌍을 입력으로 받음
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(pairs)

        # 점수와 청크를 묶어 내림차순 정렬
        scored_chunks = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in scored_chunks[:k]:
            results.append(
                RetrievedChunk(
                    content=chunk.content,
                    score=round(float(score), 4),
                    metadata={**chunk.metadata, "reranker_score": round(float(score), 4)},
                )
            )
        return results
