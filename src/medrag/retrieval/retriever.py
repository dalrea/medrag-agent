"""Dense Retriever 모듈 — ChromaDB 벡터 유사도 검색"""

from dataclasses import dataclass, field

from medrag.config import settings
from medrag.ingestion.embedder import Embedder


@dataclass
class RetrievedChunk:
    """검색된 청크와 관련도 메타데이터"""

    content: str
    score: float  # cosine similarity (0~1, 높을수록 관련도 높음)
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> int | None:
        return self.metadata.get("page")


class DenseRetriever:
    """
    ChromaDB 기반 벡터 유사도 검색기.

    쿼리를 임베딩하여 가장 유사한 청크를 top_k개 반환한다.
    score_threshold 이하 청크는 결과에서 제외하여 저품질 결과를 걸러낸다.
    """

    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder or Embedder()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or settings.retriever_top_k
        threshold = score_threshold if score_threshold is not None else settings.retriever_score_threshold

        if self._embedder.count() == 0:
            return []

        query_embedding = self._embedder.embed_query(query)
        results = self._embedder.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._embedder.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # ChromaDB cosine distance → similarity 변환 (distance = 1 - similarity)
            score = 1.0 - distance
            if score < threshold:
                continue
            chunks.append(
                RetrievedChunk(
                    content=doc,
                    score=round(score, 4),
                    metadata=meta,
                )
            )

        return sorted(chunks, key=lambda c: c.score, reverse=True)
