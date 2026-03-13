"""MedRAG Agent — Gradio Web UI"""

from pathlib import Path

import gradio as gr

from medrag.agent import MedRAGAgent

# 앱 전체에서 하나의 Agent 인스턴스 공유
_agent = MedRAGAgent()


# ── Ingestion 탭 ──────────────────────────────────────────────────────────

def ingest_files(files: list) -> str:
    if not files:
        return "파일을 선택해 주세요."

    total = {"documents": 0, "chunks": 0, "added": 0}
    for file in files:
        result = _agent.ingest(file.name)
        for key in total:
            total[key] += result[key]

    status = _agent.status()
    return (
        f"✅ Ingestion 완료\n\n"
        f"- 로드된 문서: {total['documents']}개\n"
        f"- 생성된 청크: {total['chunks']}개\n"
        f"- 새로 추가된 청크: {total['added']}개\n"
        f"- 총 인덱싱 청크 (ChromaDB): {status['indexed_chunks (ChromaDB)']}개\n"
        f"- 총 인덱싱 청크 (BM25): {status['indexed_chunks (BM25)']}개"
    )


def get_status() -> str:
    info = _agent.status()
    lines = [f"**{k}**: {v}" for k, v in info.items()]
    return "\n".join(lines)


# ── Chat 탭 ───────────────────────────────────────────────────────────────

def respond(message: str, history: list[list[str | None]]) -> tuple[str, list]:
    """Gradio Chatbot 콜백 — 메시지를 받아 답변과 업데이트된 히스토리 반환"""
    if not message.strip():
        return "", history

    status = _agent.status()
    if status["indexed_chunks (ChromaDB)"] == 0:
        reply = "⚠️ 인덱싱된 문서가 없습니다. '문서 업로드' 탭에서 먼저 문서를 ingestion하세요."
        history.append([message, reply])
        return "", history

    result = _agent.chat(message)

    # 출처 정보 포맷팅
    sources_md = ""
    if result.sources:
        lines = ["\n\n---\n**📚 참조 문서**"]
        for src in result.sources:
            page = f" p.{src['page']}" if src.get("page") else ""
            score = src["relevance_score"]
            lines.append(f"- `{src['document']}`{page} (관련도: {score:.3f})")
        sources_md = "\n".join(lines)

    # 재구성된 쿼리가 있으면 표시
    rewritten_md = ""
    if result.rewritten_query:
        rewritten_md = f"\n\n> 🔍 **검색 쿼리**: {result.rewritten_query}"

    confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
        result.confidence, "⚪"
    )
    confidence_md = f"\n\n{confidence_emoji} 신뢰도: **{result.confidence}**"

    full_reply = result.answer + rewritten_md + confidence_md + sources_md

    history.append([message, full_reply])
    return "", history


def reset_chat() -> tuple[str, list]:
    _agent.reset_chat()
    return "", []


# ── UI 레이아웃 ───────────────────────────────────────────────────────────

with gr.Blocks(
    title="MedRAG Agent",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 900px; margin: auto; }",
) as demo:
    gr.Markdown(
        """
        # 🏥 MedRAG Agent
        **의료 도메인 특화 RAG 기반 Q&A Agent**

        의료기기 문서, 임상 가이드라인, 규격서(IEC 62304 등)를 업로드하고
        자연어로 질문하면 문서 근거 기반 답변을 제공합니다.
        """
    )

    with gr.Tabs():

        # ── 탭 1: 문서 업로드 ────────────────────────────────────────────
        with gr.TabItem("📂 문서 업로드"):
            gr.Markdown("PDF, TXT, Markdown 파일을 업로드하여 인덱싱합니다.")

            with gr.Row():
                file_input = gr.File(
                    label="문서 선택 (복수 선택 가능)",
                    file_types=[".pdf", ".txt", ".md", ".markdown"],
                    file_count="multiple",
                )

            with gr.Row():
                ingest_btn = gr.Button("🚀 Ingestion 시작", variant="primary")
                status_btn = gr.Button("📊 상태 확인")

            ingest_output = gr.Textbox(
                label="결과",
                lines=8,
                interactive=False,
            )

            ingest_btn.click(fn=ingest_files, inputs=file_input, outputs=ingest_output)
            status_btn.click(fn=get_status, outputs=ingest_output)

        # ── 탭 2: Q&A 채팅 ───────────────────────────────────────────────
        with gr.TabItem("💬 Q&A 채팅"):
            gr.Markdown(
                "업로드한 문서를 기반으로 질문하세요. "
                "대화 맥락을 유지하며 멀티턴 질의응답이 가능합니다."
            )

            chatbot = gr.Chatbot(
                label="MedRAG Agent",
                height=480,
                bubble_full_width=False,
                render_markdown=True,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="질문을 입력하세요... (Enter로 전송)",
                    label="질문",
                    scale=9,
                    show_label=False,
                )
                send_btn = gr.Button("전송", variant="primary", scale=1)

            reset_btn = gr.Button("🔄 대화 초기화", variant="secondary")

            # 예시 질문
            gr.Examples(
                examples=[
                    ["IEC 62304의 소프트웨어 안전 등급 분류 기준은 무엇인가요?"],
                    ["DICOM 표준에서 SOP Class란 무엇인가요?"],
                    ["의료기기 소프트웨어 검증(Verification)과 확인(Validation)의 차이는?"],
                ],
                inputs=msg_input,
            )

            # 이벤트 연결
            msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            reset_btn.click(fn=reset_chat, outputs=[msg_input, chatbot])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
