import time

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from loguru import logger

from src import config
from src.chat.base_chat import BaseChat
from src.chat.rag_chat import RAGChat


def main(limit: int = 10):
    if not (config.PROCESSED_DIR / "TCGA_Reports_txt").exists():
        logger.error("TCGA_Reports_txt directory does not exist. Run ocr_pymupdf.py first.")
        return
    model_name = "gpt-4o"
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=model_name,
        include_reason=False
    )
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=0.7,
        model=model_name,
        include_reason=False
    )

    rag_test_cases = []
    base_test_cases = []

    count = 0

    rag_chat = RAGChat()
    base_chat = BaseChat()

    start_time = time.time()
    for report_path in (config.PROCESSED_DIR / "TCGA_Reports_txt").iterdir():
        report_name = report_path.stem
        logger.info(f"({count + 1}/{limit}) Processing {report_name}")
        report_text = report_path.read_text()

        rag_answer, rag_context = rag_chat.generate(report_text)
        base_answer = base_chat.generate(report_text)
        
        rag_test_cases.append(
            LLMTestCase(
                input=report_text,
                actual_output=rag_answer,
                retrieval_context=rag_context,
            )
        )

        base_test_cases.append(
            LLMTestCase(
                input=report_text,
                actual_output=base_answer,
            )
        )

        count += 1
        if count == limit:
            break
    end_time = time.time()
    
    rag_results = evaluate(
        test_cases=rag_test_cases,
        metrics=[answer_relevancy_metric, contextual_relevancy_metric],
    )
    base_results = evaluate(
        test_cases=base_test_cases,
        metrics=[answer_relevancy_metric],
    )

    logger.info(f"Time taken: {end_time - start_time}")

    # save results
    logger.info("Saving results...")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_DIR / "rag_results.json", "w") as f:
        f.write(rag_results.model_dump_json(indent=4))
    with open(config.REPORT_DIR / "base_results.json", "w") as f:
        f.write(base_results.model_dump_json(indent=4))

if __name__ == "__main__":
    main(10)
