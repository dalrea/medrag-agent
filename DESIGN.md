# MedRAG Agent — 설계 문서

> 의료 도메인 특화 RAG 기반 Q&A Agent
> 작성일: 2026-03-13

---

## 1. 프로젝트 개요

### 배경 및 목적

의료기기 소프트웨어 개발 현장에서는 방대한 기술 문서(사용 설명서, 규격서, 임상 가이드라인 등)를
검색하고 이해하는 데 많은 시간이 소요된다. 본 프로젝트는 이러한 도메인 특화 문서를 LLM이
정확하게 참조할 수 있도록 RAG(Retrieval-Augmented Generation) 구조를 설계하고,
실용적인 Q&A Agent를 구현하는 것을 목표로 한다.

### 핵심 가치

- **정확성**: 환각(Hallucination) 최소화를 위한 근거 문서 인용 강제
- **도메인 특화**: 의료/제조 도메인 용어와 문서 구조에 최적화된 청킹 전략
- **투명성**: 답변 근거가 되는 문서 출처와 관련도 점수를 함께 제공

---

## 2. 요구사항

### 2.1 기능 요구사항 (Functional Requirements)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| FR-01 | PDF, TXT, Markdown 형식의 문서를 ingestion하여 벡터 DB에 저장할 수 있어야 한다 | 필수 |
| FR-02 | 자연어 질문을 입력하면 관련 문서 청크를 검색하고, LLM이 답변을 생성해야 한다 | 필수 |
| FR-03 | 답변에는 참조한 문서명, 페이지/섹션, 관련도 점수가 포함되어야 한다 | 필수 |
| FR-04 | 대화 히스토리를 유지하여 멀티턴 질의응답이 가능해야 한다 | 필수 |
| FR-05 | 검색 결과가 없거나 신뢰도가 낮은 경우 "모르겠습니다"로 응답해야 한다 | 필수 |
| FR-06 | CLI 인터페이스로 질의응답을 수행할 수 있어야 한다 | 필수 |
| FR-07 | 간단한 Web UI(Gradio 또는 Streamlit)를 통해 인터랙티브하게 사용할 수 있어야 한다 | 선택 |
| FR-08 | 새 문서를 추가해도 전체 재인덱싱 없이 증분 업데이트가 가능해야 한다 | 선택 |

### 2.2 비기능 요구사항 (Non-Functional Requirements)

| ID | 요구사항 | 기준 |
|----|----------|------|
| NFR-01 | 단일 질의 응답 시간 | 10초 이내 (로컬 환경 기준) |
| NFR-02 | 문서 ingestion 처리량 | 100페이지 PDF 기준 2분 이내 |
| NFR-03 | 검색 정밀도(Precision@5) | 개발 QA셋 기준 0.7 이상 목표 |
| NFR-04 | 코드 품질 | 모듈별 단위 테스트, 커버리지 60% 이상 |
| NFR-05 | 환경 재현성 | requirements.txt 또는 pyproject.toml로 의존성 고정 |

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        MedRAG Agent                             │
│                                                                 │
│  ┌──────────────┐    ┌───────────────────────────────────────┐  │
│  │  Ingestion   │    │              Query Pipeline            │  │
│  │  Pipeline    │    │                                       │  │
│  │              │    │  User Query                           │  │
│  │  Documents   │    │      │                                │  │
│  │  (PDF/TXT/MD)│    │      ▼                                │  │
│  │      │       │    │  ┌─────────────┐                      │  │
│  │      ▼       │    │  │Query Rewriter│  (대화 히스토리 반영) │  │
│  │  Document    │    │  └──────┬──────┘                      │  │
│  │  Loader      │    │         │                              │  │
│  │      │       │    │         ▼                              │  │
│  │      ▼       │    │  ┌─────────────┐                      │  │
│  │  Text        │    │  │  Retriever  │◄──── Vector DB        │  │
│  │  Splitter    │    │  │  (Hybrid)   │     (ChromaDB)        │  │
│  │  (Chunker)   │    │  └──────┬──────┘                      │  │
│  │      │       │    │         │                              │  │
│  │      ▼       │    │         ▼                              │  │
│  │  Embedding   │    │  ┌─────────────┐                      │  │
│  │  Model       │    │  │   Reranker  │  (관련도 재정렬)       │  │
│  │      │       │    │  └──────┬──────┘                      │  │
│  │      ▼       │────►         │                              │  │
│  │  Vector DB   │    │         ▼                              │  │
│  │  (ChromaDB)  │    │  ┌─────────────┐                      │  │
│  └──────────────┘    │  │LLM Generator│  (Claude / GPT-4o)   │  │
│                      │  └──────┬──────┘                      │  │
│                      │         │                              │  │
│                      │         ▼                              │  │
│                      │    Answer + Sources                    │  │
│                      └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 컴포넌트 설계

### 4.1 Ingestion Pipeline

```
DocumentLoader
 ├── PDFLoader        (pymupdf)
 ├── TextLoader       (plain text)
 └── MarkdownLoader   (markdown-it)
         │
         ▼
TextSplitter (청킹 전략)
 ├── 청크 크기: 512 tokens
 ├── 오버랩: 64 tokens
 └── 의료 문서 특화: 섹션 제목 기준 우선 분리
         │
         ▼
EmbeddingModel
 └── sentence-transformers/all-MiniLM-L6-v2
     (또는 OpenAI text-embedding-3-small)
         │
         ▼
VectorStore (ChromaDB)
 └── 메타데이터: {source, page, section, chunk_id}
```

**설계 결정 이유:**
- ChromaDB: 로컬 실행 가능, 별도 서버 불필요 → 개발/시연 환경에서 간편
- 512 토큰 청크: 의료 문서의 단락 단위와 부합하며 컨텍스트 손실 최소화
- 오버랩 64 토큰: 청크 경계에서 문맥 단절 방지

### 4.2 Query Pipeline

```
QueryRewriter
 └── 대화 히스토리 + 현재 질문 → 독립적인 단일 질문으로 재구성
         │
         ▼
HybridRetriever
 ├── Dense Retrieval: 벡터 유사도 검색 (ChromaDB)
 └── Sparse Retrieval: BM25 키워드 검색
     → RRF(Reciprocal Rank Fusion)로 결과 통합
         │
         ▼
Reranker
 └── cross-encoder/ms-marco-MiniLM-L-6-v2
     → Top-K(5) 문서 관련도 재정렬
         │
         ▼
LLMGenerator
 └── 시스템 프롬프트:
     "제공된 컨텍스트에만 기반하여 답변하라.
      컨텍스트에 없는 내용은 '문서에서 찾을 수 없습니다'라고 답하라."
```

**설계 결정 이유:**
- Hybrid Retrieval: Dense만 쓸 경우 정확한 의료 용어/수치 검색에 취약 → BM25 병행
- Reranker: 초기 검색 결과의 순서를 교차 인코더로 재정렬하여 정밀도 향상
- 엄격한 시스템 프롬프트: 의료 도메인 특성상 환각은 치명적 → 근거 없는 생성 금지

### 4.3 Answer Format

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "IEC_62304_guideline.pdf",
      "section": "5.2 Software Development Planning",
      "page": 23,
      "relevance_score": 0.87,
      "excerpt": "..."
    }
  ],
  "confidence": "high | medium | low | unknown"
}
```

---

## 5. 기술 스택

| 레이어 | 기술 | 선택 이유 |
|--------|------|-----------|
| 언어 | Python 3.11 | 공고 필수 요건, AI/ML 생태계 |
| LLM | Claude API (claude-sonnet-4-6) | 긴 컨텍스트 처리, 정확한 instruction-following |
| Orchestration | LangChain | RAG 파이프라인 구성 표준, 풍부한 문서 |
| Vector DB | ChromaDB | 로컬 실행, 경량, 빠른 프로토타이핑 |
| Embedding | sentence-transformers | 로컬 실행 가능, 무료 |
| Reranker | cross-encoder (HuggingFace) | 검색 정밀도 향상 |
| BM25 | rank-bm25 | 키워드 기반 검색 보완 |
| PDF 파싱 | pymupdf (fitz) | 레이아웃 보존, 빠른 속도 |
| Web UI | Gradio | 빠른 프로토타이핑, ML 프로젝트 표준 |
| 테스트 | pytest | 단위 테스트 |
| 패키지 관리 | uv + pyproject.toml | 현대적 Python 패키지 관리 |

---

## 6. 디렉토리 구조 (예정)

```
medrag-agent/
├── pyproject.toml
├── README.md
├── DESIGN.md                  ← 이 문서
│
├── data/
│   ├── raw/                   # 원본 문서 (PDF, TXT, MD)
│   └── vectorstore/           # ChromaDB 저장소
│
├── src/
│   └── medrag/
│       ├── __init__.py
│       ├── ingestion/
│       │   ├── loader.py      # 문서 로더
│       │   ├── splitter.py    # 청킹
│       │   └── embedder.py    # 임베딩 + 저장
│       ├── retrieval/
│       │   ├── retriever.py   # Hybrid Retriever
│       │   └── reranker.py    # Cross-encoder Reranker
│       ├── generation/
│       │   ├── rewriter.py    # Query Rewriter
│       │   └── generator.py   # LLM Generator
│       ├── agent.py           # 전체 파이프라인 조합
│       └── config.py          # 설정값 중앙 관리
│
├── app/
│   └── gradio_app.py          # Web UI
│
├── cli/
│   └── main.py                # CLI 인터페이스
│
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    └── test_agent.py
```

---

## 7. 주요 설계 결정 및 트레이드오프

### 결정 1: Fine-tuning vs RAG
- **선택**: RAG
- **이유**: 의료 문서는 자주 업데이트됨. Fine-tuning은 재학습 비용이 높고 새 문서 반영이 느림.
  RAG는 문서 추가만으로 즉시 반영 가능하며, 출처 추적이 가능하여 신뢰성 검증에 유리.

### 결정 2: 로컬 LLM vs API LLM
- **선택**: API LLM (Claude)
- **이유**: 로컬 LLM(Ollama 등)은 하드웨어 의존성이 높고 의료 도메인 instruction-following 품질이 낮음.
  API 방식은 재현 가능하며 포트폴리오 시연에 적합. 비용 최적화는 이후 과제.

### 결정 3: LangChain vs 직접 구현
- **선택**: LangChain 기반 구현 (단, 핵심 로직은 직접 작성)
- **이유**: LangChain의 추상화를 무비판적으로 사용하면 내부 동작 이해 부족으로 보일 수 있음.
  Retriever, Reranker, Generator는 직접 클래스로 구현하고, LangChain은 파이프라인 연결에만 활용.

---

## 8. 평가 지표

| 지표 | 설명 | 측정 방법 |
|------|------|-----------|
| Precision@5 | 상위 5개 검색 결과 중 관련 문서 비율 | 수동 라벨링 QA셋 20개 |
| Answer Faithfulness | 답변이 검색된 컨텍스트에만 기반하는가 | RAGAS 라이브러리 활용 |
| Answer Relevance | 답변이 질문에 실제로 답하는가 | RAGAS 라이브러리 활용 |
| Latency (P95) | 95번째 백분위 응답 시간 | 로컬 환경 벤치마크 |

---

## 9. 단계별 구현 계획

| 단계 | 목표 | 산출물 |
|------|------|--------|
| Phase 1 (1주) | 기본 RAG 파이프라인 구현 | ingestion + dense retrieval + LLM 답변 |
| Phase 2 (1주) | 품질 향상 | Hybrid Retrieval + Reranker + 출처 인용 |
| Phase 3 (3일) | 멀티턴 + Web UI | Query Rewriter + Gradio UI |
| Phase 4 (3일) | 평가 및 문서화 | RAGAS 평가, README, 시연 영상 |

---

## 10. 포트폴리오 어필 포인트

1. **기술 선택의 근거**: 단순히 "RAG를 썼다"가 아니라, Fine-tuning 대비 RAG를 선택한 이유,
   Hybrid Retrieval을 도입한 이유 등 **트레이드오프 기반 의사결정** 능력 시연

2. **도메인 연결**: 의료기기 SW 실무 경험(DICOM, 의료 문서 구조 이해)을
   AI 시스템 설계에 직접 적용 → 삼성 DS AI센터의 "도메인 전문가 협업" 역할과 연결

3. **평가 지표 도입**: 단순 데모에 그치지 않고 RAGAS로 성능을 정량 측정 →
   "AI 모델 성능 최적화 경험" 우대사항에 직접 대응

4. **코드 품질**: 모듈 구조, 테스트 코드, 설정 중앙화 → 실무 SW 엔지니어링 역량 시연
