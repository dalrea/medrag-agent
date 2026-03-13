"""MedRAG Agent — Ingestion과 Query 파이프라인을 통합하는 진입점"""

from pathlib import Path

from medrag.config import settings
from medrag.generation.generator import GenerationResult, LLMGenerator
from medrag.ingestion.embedder import Embedder
from medrag.ingestion.loader import DocumentLoader
from medrag.ingestion.splitter import TextSplitter
from medrag.retrieval.retriever import DenseRetriever, RetrievedChunk


class MedRAGAgent:
    """
    MedRAG Agent 메인 클래스.

    - ingest(): 문서를 로드 → 청킹 → 임베딩 → ChromaDB 저장
    - query(): 질문을 받아 관련 문서 검색 → LLM으로 답변 생성
    - chat(): 대화 히스토리를 유지하는 멀티턴 인터페이스
    """

    def __init__(self):
        self._loader = DocumentLoader()
        self._splitter = TextSplitter()
        self._embedder = Embedder()
        self._retriever = DenseRetriever(embedder=self._embedder)
        self._generator = LLMGenerator()
        self._chat_history: list[dict] = []

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest(self, path: str | Path) -> dict:
        """
        단일 파일 또는 디렉토리를 ingestion.

        Returns:
            {"documents": int, "chunks": int, "added": int}
        """
        path = Path(path)
        if path.is_dir():
            docs = self._loader.load_directory(path)
        else:
            docs = self._loader.load(path)

        chunks = self._splitter.split(docs)
        added = self._embedder.add_chunks(chunks)

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
    ) -> GenerationResult:
        """단일 질의응답 (히스토리 없음)"""
        chunks = self._retriever.retrieve(question, top_k=top_k)
        return self._generator.generate(question, chunks)

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """LLM 생성 없이 검색 결과만 반환 (디버깅/평가용)"""
        return self._retriever.retrieve(question, top_k=top_k)

    # ── Chat (멀티턴) ──────────────────────────────────────────────────────

    def chat(self, message: str) -> GenerationResult:
        """
        대화 히스토리를 유지하는 멀티턴 질의응답.
        Phase 3에서 QueryRewriter와 통합 예정.
        """
        chunks = self._retriever.retrieve(message)
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

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "indexed_chunks": self._embedder.count(),
            "collection": settings.chroma_collection_name,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
        }
