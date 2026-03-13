# MedRAG Agent — Claude 작업 지침

## 프로젝트 목적
삼성 DS AI센터 SW개발 직무 지원을 위한 포트폴리오 프로젝트.
의료 도메인 특화 RAG 기반 Q&A Agent.

## 커밋 메세지 규칙
- Co-Authored-By: Claude 줄을 절대 포함하지 않는다.
- 제목은 영어, 본문은 한국어.
- 제목 50자 이내, 본문 72자 이내 줄바꿈.
- Conventional Commits 형식: feat / fix / refactor / docs / chore / test

## 작업 맥락
- 구현 설계 문서: DESIGN.md
- 단계별 계획:
  - Phase 1 (1주): 기본 RAG 파이프라인 (ingestion + dense retrieval + LLM 답변)
  - Phase 2 (1주): Hybrid Retrieval + Reranker + 출처 인용
  - Phase 3 (3일): 멀티턴 + Gradio Web UI
  - Phase 4 (3일): RAGAS 평가 + README + 시연 영상
- Remote: https://github.com/dalrea/medrag-agent.git

## 기술 스택
- Python 3.11, LangChain, ChromaDB, sentence-transformers
- LLM: Claude API (claude-sonnet-4-6)
- Web UI: Gradio
- 패키지 관리: uv + pyproject.toml

## 사용자 정보
- 현재 직장: Genoray, Modality Application 개발 (C++/Qt, 1년 8개월)
- SSAFY 7기 Python 트랙 수료, 코치 경험
- 삼성 DS AI센터 공채 지원 준비 중
