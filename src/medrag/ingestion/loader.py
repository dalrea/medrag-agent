"""문서 로더 모듈 — PDF, TXT, Markdown 형식 지원"""

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf


@dataclass
class Document:
    """로드된 문서의 기본 단위"""

    content: str
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """PDF 문서 로더 (pymupdf 기반)"""

    def load(self, path: Path) -> list[Document]:
        docs = []
        with fitz.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        content=text,
                        metadata={
                            "source": path.name,
                            "page": page_num,
                            "total_pages": len(pdf),
                            "file_type": "pdf",
                        },
                    )
                )
        return docs


class TextLoader:
    """TXT 문서 로더"""

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return []
        return [
            Document(
                content=content,
                metadata={
                    "source": path.name,
                    "file_type": "txt",
                },
            )
        ]


class MarkdownLoader:
    """Markdown 문서 로더"""

    def load(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return []
        return [
            Document(
                content=content,
                metadata={
                    "source": path.name,
                    "file_type": "markdown",
                },
            )
        ]


class DocumentLoader:
    """파일 확장자에 따라 적절한 로더를 선택하는 통합 로더"""

    _LOADERS = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
    }

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

        suffix = path.suffix.lower()
        loader_cls = self._LOADERS.get(suffix)
        if loader_cls is None:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

        return loader_cls().load(path)

    def load_directory(self, directory: str | Path) -> list[Document]:
        """디렉토리 내 지원 형식 파일을 모두 로드"""
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"디렉토리가 아닙니다: {directory}")

        docs = []
        for suffix in self._LOADERS:
            for file_path in sorted(directory.glob(f"**/*{suffix}")):
                docs.extend(self.load(file_path))
        return docs
