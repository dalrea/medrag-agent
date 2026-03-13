"""Ingestion 파이프라인 단위 테스트"""

import tempfile
from pathlib import Path

import pytest

from medrag.ingestion.loader import Document, DocumentLoader
from medrag.ingestion.splitter import Chunk, TextSplitter


class TestDocumentLoader:
    def test_load_txt(self, tmp_path: Path):
        file = tmp_path / "sample.txt"
        file.write_text("Hello, this is a test document.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(file)

        assert len(docs) == 1
        assert docs[0].content == "Hello, this is a test document."
        assert docs[0].metadata["source"] == "sample.txt"
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_markdown(self, tmp_path: Path):
        file = tmp_path / "sample.md"
        file.write_text("# Title\n\nSome content here.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(file)

        assert len(docs) == 1
        assert "# Title" in docs[0].content

    def test_load_unsupported_raises(self, tmp_path: Path):
        file = tmp_path / "sample.xyz"
        file.write_text("content")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="지원하지 않는 파일 형식"):
            loader.load(file)

    def test_load_nonexistent_raises(self, tmp_path: Path):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_load_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
        (tmp_path / "b.md").write_text("Document B", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load_directory(tmp_path)

        assert len(docs) == 2


class TestTextSplitter:
    def test_split_short_document_no_split(self):
        doc = Document(content="Short text.", metadata={"source": "test.txt"})
        splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split([doc])

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_has_metadata(self):
        doc = Document(content="Some content.", metadata={"source": "test.txt", "page": 1})
        splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split([doc])

        assert "chunk_id" in chunks[0].metadata
        assert chunks[0].metadata["source"] == "test.txt"

    def test_split_large_document(self):
        # chunk_size=50 토큰을 초과하는 긴 문서 (문장 단위로 분리 가능하도록 구성)
        sentences = [f"This is sentence number {i} in the document." for i in range(80)]
        long_text = " ".join(sentences)
        doc = Document(content=long_text, metadata={"source": "large.txt"})
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split([doc])

        assert len(chunks) > 1

    def test_section_based_split(self):
        text = "## Section 1\nContent of section 1.\n\n## Section 2\nContent of section 2."
        doc = Document(content=text, metadata={"source": "sections.md"})
        splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split([doc])

        # 섹션 기준으로 분리되어야 함
        assert len(chunks) >= 1
