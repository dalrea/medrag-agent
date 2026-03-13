"""임베딩 및 ChromaDB 저장 모듈"""

import chromadb
from sentence_transformers import SentenceTransformer

from medrag.config import settings
from medrag.ingestion.splitter import Chunk


class Embedder:
    """
    청크를 임베딩하여 ChromaDB에 저장하는 클래스.

    sentence-transformers 모델을 사용해 로컬에서 임베딩을 생성하여
    외부 API 비용 없이 벡터 DB를 구축한다.
    """

    def __init__(self):
        self._model = SentenceTransformer(settings.embedding_model)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir)
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """청크를 임베딩하여 ChromaDB에 저장. 이미 존재하는 chunk_id는 건너뜀."""
        if not chunks:
            return 0

        # 중복 제거: 이미 저장된 chunk_id 확인
        ids = [c.metadata.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
        existing = self._collection.get(ids=ids, include=[])["ids"]
        existing_set = set(existing)

        new_chunks = [
            (chunk_id, chunk)
            for chunk_id, chunk in zip(ids, chunks)
            if chunk_id not in existing_set
        ]
        if not new_chunks:
            return 0

        new_ids, new_chunk_objs = zip(*new_chunks)
        texts = [c.content for c in new_chunk_objs]
        embeddings = self._model.encode(texts, show_progress_bar=True).tolist()
        metadatas = [c.metadata for c in new_chunk_objs]

        self._collection.add(
            ids=list(new_ids),
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(new_ids)

    def embed_query(self, query: str) -> list[float]:
        """쿼리 문자열을 임베딩 벡터로 변환"""
        return self._model.encode(query).tolist()

    def count(self) -> int:
        """저장된 청크 수 반환"""
        return self._collection.count()
