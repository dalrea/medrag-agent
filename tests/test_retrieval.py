"""Retrieval 파이프라인 단위 테스트"""

import pytest

from medrag.retrieval.bm25 import BM25Index
from medrag.retrieval.hybrid import HybridRetriever, _reciprocal_rank_fusion
from medrag.retrieval.retriever import RetrievedChunk


def _make_chunk(content: str, chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        score=score,
        metadata={"chunk_id": chunk_id, "source": "test.txt"},
    )


class TestBM25Index:
    def test_search_returns_relevant_result(self, tmp_path, monkeypatch):
        # BM25 인덱스 저장 경로를 tmp_path로 교체
        import medrag.retrieval.bm25 as bm25_module
        monkeypatch.setattr(
            bm25_module,
            "_BM25_INDEX_PATH",
            tmp_path / "bm25_index.pkl",
        )

        index = BM25Index()
        texts = [
            "IEC-62304 is a medical device software standard.",
            "Python is a general-purpose programming language.",
            "FDA guidance covers software validation requirements.",
        ]
        ids = ["doc_0", "doc_1", "doc_2"]
        metadatas = [{"source": "test.txt", "chunk_id": cid} for cid in ids]
        index.add(texts, ids, metadatas)

        results = index.search("IEC-62304 medical device", top_k=3)

        assert len(results) > 0
        # IEC-62304 문서가 상위에 위치해야 함
        assert "IEC-62304" in results[0].content

    def test_empty_index_returns_empty(self, tmp_path, monkeypatch):
        import medrag.retrieval.bm25 as bm25_module
        monkeypatch.setattr(
            bm25_module,
            "_BM25_INDEX_PATH",
            tmp_path / "bm25_index.pkl",
        )

        index = BM25Index()
        results = index.search("anything", top_k=5)
        assert results == []

    def test_duplicate_ids_not_added(self, tmp_path, monkeypatch):
        import medrag.retrieval.bm25 as bm25_module
        monkeypatch.setattr(
            bm25_module,
            "_BM25_INDEX_PATH",
            tmp_path / "bm25_index.pkl",
        )

        index = BM25Index()
        texts = ["Medical software standard."]
        ids = ["doc_0"]
        meta = [{"source": "test.txt", "chunk_id": "doc_0"}]

        index.add(texts, ids, meta)
        index.add(texts, ids, meta)  # 중복 추가

        assert index.count() == 1

    def test_tokenize_preserves_hyphenated_terms(self):
        tokens = BM25Index._tokenize("IEC-62304 COVID-19 reference")
        assert "iec-62304" in tokens
        assert "covid-19" in tokens


class TestRRF:
    def test_rrf_merges_two_lists(self):
        list1 = [
            _make_chunk("doc A", "a"),
            _make_chunk("doc B", "b"),
        ]
        list2 = [
            _make_chunk("doc B", "b"),
            _make_chunk("doc C", "c"),
        ]
        result = _reciprocal_rank_fusion([list1, list2])

        ids = [c.metadata["chunk_id"] for c in result]
        # doc B는 두 리스트 모두에 있으므로 가장 높은 RRF 점수
        assert ids[0] == "b"

    def test_rrf_single_list_passthrough(self):
        chunks = [_make_chunk(f"doc {i}", str(i)) for i in range(3)]
        result = _reciprocal_rank_fusion([chunks])
        assert len(result) == 3

    def test_rrf_score_stored_in_metadata(self):
        chunks = [_make_chunk("doc", "x")]
        result = _reciprocal_rank_fusion([chunks])
        assert "rrf_score" in result[0].metadata
