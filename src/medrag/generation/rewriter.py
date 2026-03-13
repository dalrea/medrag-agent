"""Query Rewriter — 대화 히스토리를 반영한 독립적 질의 재구성"""

import anthropic

from medrag.config import settings

_REWRITE_SYSTEM_PROMPT = """당신은 대화 문맥을 분석하여 질문을 재구성하는 전문가입니다.

다음 규칙을 따르세요:
1. 이전 대화 히스토리와 현재 질문을 보고, 현재 질문이 문서 검색에 최적화된
   독립적인 단일 질문이 되도록 재구성하세요.
2. 대화 맥락 없이도 의미가 통하도록, 대명사(그것, 이것, 해당 등)를
   구체적인 명사로 치환하세요.
3. 질문의 핵심 의도는 반드시 유지하세요.
4. 재구성된 질문만 출력하고, 다른 설명은 일절 포함하지 마세요.
5. 히스토리가 없거나 현재 질문이 이미 독립적이면 원문 그대로 반환하세요."""


class QueryRewriter:
    """
    멀티턴 대화에서 현재 질문을 독립적인 검색 쿼리로 재구성.

    예시:
      히스토리: "IEC 62304가 무엇인가요?" → "의료기기 소프트웨어 수명주기 표준입니다."
      현재 질문: "그 표준의 클래스 분류 기준은 뭔가요?"
      재구성 결과: "IEC 62304 의료기기 소프트웨어 클래스 분류 기준은 무엇인가요?"

    이 재구성 덕분에 Retriever가 대화 맥락 없이도 올바른 문서를 찾을 수 있다.
    """

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def rewrite(self, query: str, chat_history: list[dict]) -> str:
        """
        대화 히스토리를 반영하여 질의를 재구성.

        Args:
            query: 현재 사용자 질문
            chat_history: [{"role": "user"|"assistant", "content": str}, ...]

        Returns:
            검색에 최적화된 독립적 질문 문자열
        """
        if not chat_history:
            return query

        # 히스토리를 읽기 쉬운 텍스트로 변환 (최근 6턴만 사용)
        recent_history = chat_history[-12:]
        history_text = self._format_history(recent_history)

        response = self._client.messages.create(
            model=settings.llm_model,
            max_tokens=256,
            temperature=0.0,
            system=_REWRITE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"[이전 대화]\n{history_text}\n\n"
                        f"[현재 질문]\n{query}\n\n"
                        "위 대화 맥락을 반영하여 현재 질문을 독립적인 검색 쿼리로 재구성하세요."
                    ),
                }
            ],
        )

        rewritten = response.content[0].text.strip()
        return rewritten if rewritten else query

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        lines = []
        for turn in history:
            role = "사용자" if turn["role"] == "user" else "어시스턴트"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)
