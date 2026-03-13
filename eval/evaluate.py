"""
RAGAS 기반 RAG 파이프라인 평가 스크립트

평가 지표:
- Faithfulness:     답변이 검색된 컨텍스트에만 기반하는가 (환각 측정)
- Answer Relevance: 답변이 질문에 실제로 답하는가
- Context Precision: 검색된 컨텍스트 중 관련 있는 비율

사용법:
  python eval/evaluate.py --qa-file eval/qa_samples.json --output eval/results.json
"""

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from medrag.agent import MedRAGAgent


def load_qa_samples(path: str) -> list[dict]:
    """
    QA 샘플 파일 로드.

    파일 형식 (JSON):
    [
        {
            "question": "IEC 62304 클래스 분류 기준은?",
            "ground_truth": "IEC 62304는 소프트웨어를 A, B, C 세 등급으로 분류한다..."
        },
        ...
    ]
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(agent: MedRAGAgent, samples: list[dict]) -> list[dict]:
    """각 QA 샘플에 대해 RAG 파이프라인 실행, RAGAS 입력 형식으로 변환"""
    results = []
    for i, sample in enumerate(samples):
        question = sample["question"]
        ground_truth = sample.get("ground_truth", "")

        print(f"  [{i+1}/{len(samples)}] {question[:60]}...")

        # 검색 결과와 답변 동시 획득
        chunks = agent.retrieve_only(question, rerank=True)
        contexts = [c.content for c in chunks] if chunks else [""]

        gen_result = agent.query(question)

        results.append({
            "question": question,
            "answer": gen_result.answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })
    return results


def build_ragas_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "question":     [r["question"] for r in records],
        "answer":       [r["answer"] for r in records],
        "contexts":     [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })


def main():
    parser = argparse.ArgumentParser(description="MedRAG RAGAS 평가")
    parser.add_argument(
        "--qa-file",
        default="eval/qa_samples.json",
        help="평가용 QA 샘플 JSON 파일 경로",
    )
    parser.add_argument(
        "--output",
        default="eval/results.json",
        help="평가 결과 저장 경로",
    )
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"[오류] QA 샘플 파일을 찾을 수 없습니다: {qa_path}")
        print("  eval/qa_samples.json.example을 참고하여 샘플 파일을 작성하세요.")
        sys.exit(1)

    print("=== MedRAG Agent RAGAS 평가 ===\n")

    # 1. Agent 초기화
    print("[1/4] Agent 초기화 중...")
    agent = MedRAGAgent()
    status = agent.status()
    indexed = status["indexed_chunks (ChromaDB)"]
    if indexed == 0:
        print("[경고] 인덱싱된 문서가 없습니다. 먼저 문서를 ingestion하세요.")
        sys.exit(1)
    print(f"  인덱싱된 청크: {indexed}개\n")

    # 2. QA 샘플 로드
    print("[2/4] QA 샘플 로드 중...")
    samples = load_qa_samples(qa_path)
    print(f"  샘플 수: {len(samples)}개\n")

    # 3. 파이프라인 실행
    print("[3/4] RAG 파이프라인 실행 중...")
    records = run_pipeline(agent, samples)
    print()

    # 4. RAGAS 평가
    print("[4/4] RAGAS 평가 실행 중...")
    dataset = build_ragas_dataset(records)

    # RAGAS는 내부적으로 LLM을 사용해 평가 — Anthropic 모델 사용
    from langchain_anthropic import ChatAnthropic
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from medrag.config import settings

    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )

    score = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    # 5. 결과 출력 및 저장
    score_dict = score.to_pandas().mean(numeric_only=True).to_dict()

    print("\n=== 평가 결과 ===")
    print(f"  Faithfulness      (환각 방지): {score_dict.get('faithfulness', 'N/A'):.4f}")
    print(f"  Answer Relevance  (답변 관련성): {score_dict.get('answer_relevancy', 'N/A'):.4f}")
    print(f"  Context Precision (검색 정밀도): {score_dict.get('context_precision', 'N/A'):.4f}")

    output_data = {
        "summary": score_dict,
        "samples": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts_count": len(r["contexts"]),
            }
            for r in records
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
