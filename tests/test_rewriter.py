"""QueryRewriter 단위 테스트 (API 호출 없이 mock 사용)"""

from unittest.mock import MagicMock, patch

import pytest

from medrag.generation.rewriter import QueryRewriter


class TestQueryRewriter:
    def test_empty_history_returns_original(self):
        """히스토리 없으면 재구성 API 호출 없이 원문 그대로 반환"""
        rewriter = QueryRewriter()
        result = rewriter.rewrite("IEC 62304란 무엇인가요?", chat_history=[])
        assert result == "IEC 62304란 무엇인가요?"

    def test_rewrite_called_with_history(self):
        """히스토리가 있으면 Claude API를 호출하여 재구성"""
        rewriter = QueryRewriter()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="IEC 62304 소프트웨어 클래스 분류 기준은?")]

        with patch.object(rewriter._client.messages, "create", return_value=mock_response) as mock_create:
            history = [
                {"role": "user", "content": "IEC 62304가 무엇인가요?"},
                {"role": "assistant", "content": "의료기기 소프트웨어 수명주기 표준입니다."},
            ]
            result = rewriter.rewrite("그 표준의 클래스 분류 기준은 뭔가요?", history)

        assert mock_create.called
        assert result == "IEC 62304 소프트웨어 클래스 분류 기준은?"

    def test_empty_api_response_falls_back_to_original(self):
        """API가 빈 문자열을 반환하면 원문 그대로 사용"""
        rewriter = QueryRewriter()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="")]

        with patch.object(rewriter._client.messages, "create", return_value=mock_response):
            history = [{"role": "user", "content": "이전 질문"}]
            result = rewriter.rewrite("현재 질문", history)

        assert result == "현재 질문"

    def test_history_truncated_to_recent_12(self):
        """히스토리는 최근 12개 항목(6턴)만 전달됨"""
        rewriter = QueryRewriter()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="재구성된 쿼리")]

        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]

        with patch.object(rewriter._client.messages, "create", return_value=mock_response) as mock_create:
            rewriter.rewrite("현재 질문", long_history)

        call_args = mock_create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        # 최근 12개만 포함되어야 하므로 msg 8~19만 있어야 함
        assert "msg 0" not in user_content
        assert "msg 8" in user_content

    def test_format_history(self):
        history = [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"},
        ]
        formatted = QueryRewriter._format_history(history)
        assert "사용자: 질문" in formatted
        assert "어시스턴트: 답변" in formatted
