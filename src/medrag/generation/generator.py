"""LLM 답변 생성 모듈 — Claude API 기반"""

from dataclasses import dataclass, field

import anthropic

from medrag.config import settings
from medrag.retrieval.retriever import RetrievedChunk

# 시스템 프롬프트: 의료 도메인 환각 방지를 위한 엄격한 지침
_SYSTEM_PROMPT = """당신은 의료 문서 전문 Q&A 어시스턴트입니다.

반드시 다음 규칙을 따르세요:
1. 제공된 [참조 문서] 내용에만 기반하여 답변하세요.
2. 참조 문서에 없는 내용은 추측하거나 생성하지 마세요.
3. 답변할 수 없는 경우 "제공된 문서에서 해당 내용을 찾을 수 없습니다."라고 명확히 밝히세요.
4. 답변 마지막에는 반드시 참조한 문서 출처를 명시하세요.
5. 한국어로 답변하되, 의료 전문 용어는 원어(영어)를 병기하세요."""


@dataclass
class GenerationResult:
    """LLM 생성 결과"""

    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: str = "unknown"  # high | medium | low | unknown


class LLMGenerator:
    """
    Claude API를 사용한 답변 생성기.

    검색된 청크를 컨텍스트로 주입하여 근거 기반 답변을 생성한다.
    참조 문서가 없거나 신뢰도가 낮은 경우 명시적으로 응답한다.
    """

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        chat_history: list[dict] | None = None,
    ) -> GenerationResult:
        if not context_chunks:
            return GenerationResult(
                answer="제공된 문서에서 해당 내용을 찾을 수 없습니다. 다른 키워드로 질문해 주세요.",
                sources=[],
                confidence="unknown",
            )

        context_text = self._build_context(context_chunks)
        messages = self._build_messages(query, context_text, chat_history or [])
        confidence = self._estimate_confidence(context_chunks)

        response = self._client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )

        answer = response.content[0].text
        sources = self._extract_sources(context_chunks)

        return GenerationResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source_info = f"{chunk.source}"
            if chunk.page:
                source_info += f" (p.{chunk.page})"
            parts.append(f"[참조 문서 {i}] 출처: {source_info}\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _build_messages(
        self,
        query: str,
        context: str,
        history: list[dict],
    ) -> list[dict]:
        messages = list(history)
        messages.append({
            "role": "user",
            "content": f"[참조 문서]\n{context}\n\n[질문]\n{query}",
        })
        return messages

    def _extract_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        seen = set()
        sources = []
        for chunk in chunks:
            key = (chunk.source, chunk.page)
            if key in seen:
                continue
            seen.add(key)
            source = {
                "document": chunk.source,
                "relevance_score": chunk.score,
                "excerpt": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            }
            if chunk.page:
                source["page"] = chunk.page
            sources.append(source)
        return sources

    def _estimate_confidence(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "unknown"
        top_score = chunks[0].score
        if top_score >= 0.75:
            return "high"
        elif top_score >= 0.5:
            return "medium"
        else:
            return "low"
