from pathlib import Path

from rag_knowledge_system.evaluation import (
    evaluate_retriever,
    load_cases,
)
from rag_knowledge_system.pipeline import (
    build_retriever_from_file_with_paragraphs,
)


CORPUS_PATH = Path("eval/corpus/eval_rag_basics.txt")
CASES_PATH = Path("eval/retrieval_cases.json")


def main() -> None:
    retriever = build_retriever_from_file_with_paragraphs(
        file_path=str(CORPUS_PATH),
        max_words=100,
        overlap=0,
    )

    cases = load_cases(CASES_PATH)

    scores = evaluate_retriever(
        retriever=retriever,
        cases=cases,
        k=3,
    )

    print("Retrieval Evaluation")
    print("--------------------")

    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")


if __name__ == "__main__":
    main()
