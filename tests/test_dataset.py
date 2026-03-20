import os

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import HallucinationMetric
from deepeval import assert_test, evaluate
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.evaluate.utils import aggregate_metric_pass_rates
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric


class FakeMetric1(BaseMetric):
    def __init__(self, threshold: float = 0.5, _success: bool = True):
        self.threshold = threshold
        self.success = _success

    def measure(self, test_case: LLMTestCase):
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This async metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


class FakeMetric2(BaseMetric):
    def __init__(self, threshold: float = 0.5, _success: bool = True):
        self.threshold = threshold
        self.success = _success

    def measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This metric looking good!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.score = 0.5
        self.reason = "This async metric looking good!"
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Fake"


def test_create_dataset():
    dataset = EvaluationDataset()
    module_b_dir = os.path.dirname(os.path.realpath(__file__))

    file_path = os.path.join(module_b_dir, "data", "dataset.csv")

    dataset.add_test_cases_from_csv_file(
        file_path,
        input_col_name="query",
        actual_output_col_name="actual_output",
        expected_output_col_name="expected_output",
        context_col_name="context",
        retrieval_context_col_name="retrieval",
    )
    assert len(dataset.test_cases) == 5, "Test Cases not loaded from CSV"
    file_path = os.path.join(module_b_dir, "data", "dataset.json")
    dataset.add_test_cases_from_json_file(
        file_path,
        input_key_name="query",
        expected_output_key_name="expected_output",
        context_key_name="context",
        retrieval_context_key_name="retrieval",
        actual_output_key_name="actual_output",
    )
    assert len(dataset.test_cases) == 10, "Test Cases not loaded from JSON"


def test_create_dataset_from_hugging_face_dataset_rows():
    rows = [
        {
            "prompt": "What is 2 + 2?",
            "response": "4",
            "expected": "4",
            "context": ["basic arithmetic"],
            "retrieval_context": "math;addition",
            "tools_called": [{"name": "calculator"}],
            "expected_tools": ["calculator"],
            "metadata": {"source": "unit-test"},
        }
    ]

    dataset = EvaluationDataset()
    dataset.add_test_cases_from_hugging_face_dataset(
        rows,
        input_key_name="prompt",
        actual_output_key_name="response",
        expected_output_key_name="expected",
        context_key_name="context",
        retrieval_context_key_name="retrieval_context",
        tools_called_key_name="tools_called",
        expected_tools_key_name="expected_tools",
        additional_metadata_key_name="metadata",
    )

    assert len(dataset.test_cases) == 1
    test_case = dataset.test_cases[0]
    assert test_case.input == "What is 2 + 2?"
    assert test_case.actual_output == "4"
    assert test_case.expected_output == "4"
    assert test_case.context == ["basic arithmetic"]
    assert test_case.retrieval_context == ["math", "addition"]
    assert test_case.tools_called == [ToolCall(name="calculator")]
    assert test_case.expected_tools == [ToolCall(name="calculator")]
    assert test_case.additional_metadata == {"source": "unit-test"}


def test_create_dataset_from_hugging_face_dataset_rows_without_outputs():
    rows = [{"prompt": "Summarize the article."}]

    dataset = EvaluationDataset()
    dataset.add_test_cases_from_hugging_face_dataset(
        rows,
        input_key_name="prompt",
    )

    assert len(dataset.test_cases) == 1
    assert dataset.test_cases[0].actual_output == ""


def test_create_goldens_from_hugging_face_dataset_rows():
    rows = [
        {
            "prompt": "List two prime numbers.",
            "expected": "2 and 3",
            "context": "numbers;primes",
            "source": "synthetic.csv",
            "tools_called": [{"name": "calculator"}],
            "metadata": {"split": "test"},
        }
    ]

    dataset = EvaluationDataset()
    dataset.add_goldens_from_hugging_face_dataset(
        rows,
        input_key_name="prompt",
        expected_output_key_name="expected",
        context_key_name="context",
        source_file_key_name="source",
        tools_called_key_name="tools_called",
        additional_metadata_key_name="metadata",
    )

    assert len(dataset.goldens) == 1
    golden = dataset.goldens[0]
    assert isinstance(golden, Golden)
    assert golden.input == "List two prime numbers."
    assert golden.expected_output == "2 and 3"
    assert golden.context == ["numbers", "primes"]
    assert golden.source_file == "synthetic.csv"
    assert golden.tools_called == [ToolCall(name="calculator")]
    assert golden.additional_metadata == {"split": "test"}
