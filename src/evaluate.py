from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase


def evaluate(query: str, actual_output: str, retrieval_context: list[str] = None):
    metric = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-3.5-turbo",
        include_reason=True
    )

    if retrieval_context is None:
        test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        )
    else:
        test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
        )

    metric.measure(test_case)
    return metric.score, metric.reason