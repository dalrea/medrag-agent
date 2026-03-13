"""BM25 키워드 검색 모듈 — Dense 검색의 용어 매칭 약점을 보완"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from medrag.config import settings
from medrag.retrieval.retriever import RetrievedChunk

# BM25 인덱스 저장 경로
_BM25_INDEX_PATH = Path(settings.chroma_persist_dir) / "bm25_index.pkl"


class BM25Index:
    """
    BM25 역색인 관리 클래스.

    의료 문서에서 자주 등장하는 정확한 용어(약품명, 수치, 규격 코드 등)는
    Dense 임베딩 검색만으로는 놓치기 쉽다. BM25는 이러한 키워드 정확도를 보완한다.

    인덱스는 ChromaDB와 동일 경로에 pickle로 저장하여 재시작 시 재사용한다.
    """

    def __init__(self):
        self._corpus: list[str] = []       # 원본 텍스트
        self._chunk_ids: list[str] = []    # ChromaDB chunk_id와 연결
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._load_if_exists()

    # ── 인덱스 구축 ────────────────────────────────────────────────────────

    def add(self, texts: list[str], chunk_ids: list[str], metadatas: list[dict]) -> None:
        """새 문서를 인덱스에 추가하고 디스크에 저장"""
        # 중복 chunk_id 제거
        existing_ids = set(self._chunk_ids)
        new_entries = [
            (text, cid, meta)
            for text, cid, meta in zip(texts, chunk_ids, metadatas)
            if cid not in existing_ids
        ]
        if not new_entries:
            return

        for text, cid, meta in new_entries:
            self._corpus.append(text)
            self._chunk_ids.append(cid)
            self._metadatas.append(meta)

        self._rebuild()
        self._save()

    def _rebuild(self) -> None:
        tokenized = [self._tokenize(text) for text in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    # ── 검색 ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._bm25 is None or not self._corpus:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # 상위 top_k 인덱스 추출
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # BM25 점수를 0~1로 정규화
        max_score = max(scores[i] for i in top_indices) if top_indices else 1.0
        if max_score == 0:
            return []

        results = []
        for idx in top_indices:
            raw_score = scores[idx]
            if raw_score <= 0:
                continue
            normalized = raw_score / max_score
            results.append(
                RetrievedChunk(
                    content=self._corpus[idx],
                    score=round(normalized, 4),
                    metadata=self._metadatas[idx],
                )
            )
        return results

    def count(self) -> int:
        return len(self._corpus)

    # ── 영속성 ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        _BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BM25_INDEX_PATH, "wb") as f:
            pickle.dump(
                {
                    "corpus": self._corpus,
                    "chunk_ids": self._chunk_ids,
                    "metadatas": self._metadatas,
                },
                f,
            )

    def _load_if_exists(self) -> None:
        if not _BM25_INDEX_PATH.exists():
            return
        with open(_BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._corpus = data["corpus"]
        self._chunk_ids = data["chunk_ids"]
        self._metadatas = data["metadatas"]
        if self._corpus:
            self._rebuild()

    # ── 토크나이저 ────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        소문자 변환 + 알파숫자 토크나이징.
        의료 용어는 하이픈(-) 유지 (예: IEC-62304, COVID-19).
        """
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
        return tokens if tokens else [""]
