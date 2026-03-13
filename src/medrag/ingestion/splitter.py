"""텍스트 청킹 모듈 — 의료 문서 특화 청킹 전략"""

import re
from dataclasses import dataclass, field

from medrag.config import settings
from medrag.ingestion.loader import Document


@dataclass
class Chunk:
    """청킹된 텍스트 단위"""

    content: str
    metadata: dict = field(default_factory=dict)


class TextSplitter:
    """
    의료 문서 특화 텍스트 스플리터.

    전략:
    1. 마크다운/문서 섹션 제목(#, 숫자.숫자 형식) 기준으로 1차 분리
    2. 섹션이 chunk_size를 초과하면 문장 단위로 2차 분리
    3. chunk_overlap으로 청크 간 문맥 연결
    """

    # 섹션 구분 패턴: "## 제목", "1.2 제목", "Chapter N" 등
    _SECTION_PATTERN = re.compile(
        r"(?m)^(?:#{1,4}\s+.+|(?:\d+\.)+\d*\s+\S.{0,80}|Chapter\s+\d+.*)$"
    )

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            chunks.extend(self._split_document(doc))
        return chunks

    def _split_document(self, doc: Document) -> list[Chunk]:
        """문서를 섹션 단위로 분리 후 크기 초과 시 재분리"""
        sections = self._split_by_section(doc.content)
        chunks = []
        for section_idx, section_text in enumerate(sections):
            if self._token_count(section_text) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=section_text.strip(),
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{doc.metadata.get('source', '')}_{section_idx}",
                            "section_idx": section_idx,
                        },
                    )
                )
            else:
                sub_chunks = self._split_by_overlap(section_text)
                for sub_idx, text in enumerate(sub_chunks):
                    chunks.append(
                        Chunk(
                            content=text.strip(),
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata.get('source', '')}_{section_idx}_{sub_idx}",
                                "section_idx": section_idx,
                                "sub_chunk_idx": sub_idx,
                            },
                        )
                    )
        return [c for c in chunks if c.content]

    def _split_by_section(self, text: str) -> list[str]:
        """섹션 제목 기준으로 텍스트 분리"""
        matches = list(self._SECTION_PATTERN.finditer(text))
        if not matches:
            return [text]

        sections = []
        prev_end = 0
        for match in matches:
            before = text[prev_end:match.start()].strip()
            if before:
                sections.append(before)
            prev_end = match.start()
        sections.append(text[prev_end:].strip())
        return [s for s in sections if s]

    def _split_by_overlap(self, text: str) -> list[str]:
        """문장 단위로 분리 후 overlap 적용"""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = self._token_count(sentence)
            if current_len + sentence_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                # overlap: 마지막 문장들을 다음 청크 시작으로 재사용
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current):
                    overlap_tokens += self._token_count(s)
                    if overlap_tokens >= self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                current = overlap_sentences
                current_len = sum(self._token_count(s) for s in current)

            current.append(sentence)
            current_len += sentence_len

        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _token_count(text: str) -> int:
        """단어 수 기반 근사 토큰 수 계산 (영어 기준 ~1.3 토큰/단어)"""
        return int(len(text.split()) * 1.3)
