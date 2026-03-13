"""MedRAG Agent — Ingestion과 Query 파이프라인을 통합하는 진입점"""

from pathlib import Path

from medrag.config import settings
from medrag.generation.generator import GenerationResult, LLMGenerator
from medrag.ingestion.embedder import Embedder
from medrag.ingestion.loader import DocumentLoader
from medrag.ingestion.splitter import TextSplitter
from medrag.retrieval.hybrid import HybridRetriever
from medrag.retrieval.reranker import Reranker
from medrag.retrieval.retriever import RetrievedChunk


class MedRAGAgent:
    """
    MedRAG Agent 메인 클래스.

    Phase 2 파이프라인:
      ingest:  DocumentLoader → TextSplitter → Embedder(ChromaDB) + BM25Index
      query:   HybridRetriever(Dense + BM25 → RRF) → Reranker → LLMGenerator
    """

    def __init__(self):
        self._loader = DocumentLoader()
        self._splitter = TextSplitter()
        self._embedder = Embedder()
        self._retriever = HybridRetriever(embedder=self._embedder)
        self._reranker = Reranker()
        self._generator = LLMGenerator()
        self._chat_history: list[dict] = []

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest(self, path: str | Path) -> dict:
        """
        단일 파일 또는 디렉토리를 ingestion.

        - 문서 로드 → 청킹 → ChromaDB(Dense) + BM25 인덱스에 동시 저장
        Returns:
            {"documents": int, "chunks": int, "added": int}
        """
        path = Path(path)
        if path.is_dir():
            docs = self._loader.load_directory(path)
        else:
            docs = self._loader.load(path)

        chunks = self._splitter.split(docs)

        # ChromaDB 저장
        added = self._embedder.add_chunks(chunks)

        # BM25 인덱스 동기화 (새로 추가된 청크만)
        texts = [c.content for c in chunks]
        ids = [c.metadata.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
        metadatas = [c.metadata for c in chunks]
        self._retriever.bm25.add(texts, ids, metadatas)

        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "added": added,
        }

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> GenerationResult:
        """단일 질의응답 (히스토리 없음)"""
        chunks = self._retrieve_and_rerank(question, top_k=top_k, rerank=rerank)
        return self._generator.generate(question, chunks)

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """LLM 생성 없이 검색 결과만 반환 (디버깅/평가용)"""
        return self._retrieve_and_rerank(question, top_k=top_k, rerank=rerank)

    # ── Chat (멀티턴) ──────────────────────────────────────────────────────

    def chat(self, message: str) -> GenerationResult:
        """
        대화 히스토리를 유지하는 멀티턴 질의응답.
        Phase 3에서 QueryRewriter와 통합 예정.
        """
        chunks = self._retrieve_and_rerank(message)
        result = self._generator.generate(message, chunks, self._chat_history)

        # 히스토리 업데이트
        self._chat_history.append({"role": "user", "content": message})
        self._chat_history.append({"role": "assistant", "content": result.answer})

        # 히스토리 길이 제한 (최근 10턴 유지)
        if len(self._chat_history) > 20:
            self._chat_history = self._chat_history[-20:]

        return result

    def reset_chat(self) -> None:
        """대화 히스토리 초기화"""
        self._chat_history = []

    # ── 내부 유틸 ─────────────────────────────────────────────────────────

    def _retrieve_and_rerank(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        """Hybrid Retrieval → (선택) Reranking"""
        candidates = self._retriever.retrieve(query, top_k=top_k)
        if rerank and candidates:
            return self._reranker.rerank(query, candidates)
        return candidates

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "indexed_chunks (ChromaDB)": self._embedder.count(),
            "indexed_chunks (BM25)": self._retriever.bm25.count(),
            "collection": settings.chroma_collection_name,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
        }
