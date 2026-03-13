# MedRAG Agent

> 의료 도메인 특화 RAG(Retrieval-Augmented Generation) 기반 Q&A Agent

의료기기 문서, 임상 가이드라인, 규격서(IEC 62304, DICOM 등)를 인덱싱하여
자연어 질문에 **근거 문서 인용 기반**으로 답변하는 AI Agent입니다.

---

## 아키텍처

```
문서 입력 (PDF / TXT / MD)
  └─► DocumentLoader → TextSplitter (섹션 기준 청킹)
            └─► Embedder ──► ChromaDB (Dense Index)
                         └─► BM25Index (Sparse Index)

질문 입력 (멀티턴)
  └─► QueryRewriter  ← 대화 히스토리 반영, 독립 쿼리 재구성
          └─► HybridRetriever  Dense(ChromaDB) + BM25 → RRF 통합
                  └─► Reranker  cross-encoder 관련도 재정렬
                          └─► LLMGenerator (Claude)
                                  └─► 답변 + 출처 + 신뢰도
```

### 핵심 설계 결정

| 결정 | 선택 | 이유 |
|------|------|------|
| RAG vs Fine-tuning | **RAG** | 문서 업데이트 시 재학습 불필요, 출처 추적 가능 |
| Dense vs Hybrid | **Hybrid(Dense + BM25)** | 의료 코드·수치 등 키워드 정밀도 보완 |
| 검색 후처리 | **Cross-encoder Reranker** | Bi-encoder의 세밀한 관련도 포착 한계 보완 |
| LLM | **Claude API** | 긴 컨텍스트, 엄격한 instruction-following |
| 멀티턴 | **QueryRewriter** | 대명사·지시어를 구체 명사로 치환해 검색 정확도 향상 |

---

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -e .

# 환경 변수 설정
cp .env.example .env
# .env에서 ANTHROPIC_API_KEY 입력
```

### 2. 문서 Ingestion

```bash
# 단일 파일
medrag ingest data/raw/iec62304.pdf

# 디렉토리 전체
medrag ingest data/raw/
```

### 3. 질의응답

```bash
# 단일 질문
medrag ask "IEC 62304 소프트웨어 안전 등급 분류 기준은?"

# 대화형 멀티턴 모드
medrag chat

# 상태 확인
medrag status
```

### 4. Web UI 실행

```bash
python app/gradio_app.py
# http://localhost:7860 접속
```

---

## 프로젝트 구조

```
medrag-agent/
├── src/medrag/
│   ├── config.py                  # 설정 중앙 관리
│   ├── agent.py                   # 전체 파이프라인 통합
│   ├── ingestion/
│   │   ├── loader.py              # PDF / TXT / MD 로더
│   │   ├── splitter.py            # 섹션 기준 청킹 + overlap
│   │   └── embedder.py            # sentence-transformers + ChromaDB
│   ├── retrieval/
│   │   ├── retriever.py           # Dense Retriever (ChromaDB)
│   │   ├── bm25.py                # BM25 키워드 검색
│   │   ├── hybrid.py              # RRF 기반 Hybrid Retriever
│   │   └── reranker.py            # Cross-encoder Reranker
│   └── generation/
│       ├── rewriter.py            # Query Rewriter (멀티턴)
│       └── generator.py           # LLM Generator (Claude API)
├── app/
│   └── gradio_app.py              # Gradio Web UI
├── eval/
│   ├── evaluate.py                # RAGAS 평가 스크립트
│   └── qa_samples.json.example    # 평가 샘플 예시
├── tests/                         # 단위 테스트 (21개)
├── data/
│   ├── raw/                       # 원본 문서 (git 제외)
│   └── vectorstore/               # ChromaDB + BM25 인덱스 (git 제외)
├── pyproject.toml
└── .env.example
```

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.11+ |
| LLM | Claude API (`claude-sonnet-4-6`) |
| 벡터 DB | ChromaDB |
| 임베딩 | sentence-transformers (`all-MiniLM-L6-v2`) |
| 키워드 검색 | rank-bm25 |
| Reranker | cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Web UI | Gradio |
| 평가 | RAGAS |
| 패키지 관리 | pip + pyproject.toml |

---

## 평가 방법

### RAGAS 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **Faithfulness** | 답변이 검색 컨텍스트에만 기반하는가 (환각 방지) | ≥ 0.80 |
| **Answer Relevance** | 답변이 질문에 실제로 답하는가 | ≥ 0.75 |
| **Context Precision** | 검색 결과 중 관련 문서 비율 | ≥ 0.70 |

### 평가 실행

```bash
# QA 샘플 준비
cp eval/qa_samples.json.example eval/qa_samples.json
# qa_samples.json에 평가할 질문과 정답 입력

# 평가 실행
python eval/evaluate.py --qa-file eval/qa_samples.json --output eval/results.json
```

---

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API 키 | 필수 |
| `LLM_MODEL` | 사용할 Claude 모델 | `claude-sonnet-4-6` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | 청크 토큰 크기 | `512` |
| `CHUNK_OVERLAP` | 청크 간 오버랩 토큰 수 | `64` |
| `RETRIEVER_TOP_K` | Retriever 반환 청크 수 | `10` |
| `RERANKER_TOP_K` | Reranker 최종 청크 수 | `5` |

---

## 개발

```bash
# 개발 의존성 포함 설치
pip install -e ".[dev]"

# 테스트 실행
pytest tests/ -v

# 린트
ruff check src/
```
