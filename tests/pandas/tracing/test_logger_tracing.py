import datetime
import random
import sys
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from arize.pandas.logger import Client
from requests import Response


class MockResponse(Response):
    def __init__(self, df, reason, status_code):
        super().__init__()
        self.df = df
        self.reason = reason
        self.status_code = status_code


class NoSendClient(Client):
    def _post_file(self, path, sync, timeout):
        return MockResponse(pa.ipc.open_stream(pa.OSFile(path)).read_pandas(), "Success", 200)


def generate_mock_data(n) -> pd.DataFrame:
    # Helper functions for generating mock data
    def random_timestamps(n):
        start_times = [
            datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 365))
            for _ in range(n)
        ]

        end_times = [
            start + datetime.timedelta(seconds=random.randint(1, 60)) for start in start_times
        ]
        return start_times, end_times

    def mock_list_dicts(n, keys, list_length=2):
        return [
            [
                {
                    key: (
                        f"{key}_{i}_{j}"
                        if key != "attributes"
                        else {"attr_key": f"attr_value_{i}_{j}"}
                    )
                    for key in keys
                }
                for j in range(list_length)
            ]
            for i in range(n)
        ]

    def mock_dict(n, keys) -> List[Dict]:
        return [{key: f"{key}_{i}" for key in keys} for i in range(n)]

    # Generating mock data
    df = pd.DataFrame(
        {
            "context.trace_id": [str(uuid.uuid4()) for _ in range(n)],
            "context.span_id": [str(uuid.uuid4()) for _ in range(n)],
            "parent_id": [
                str(uuid.uuid4()) if i % 2 == 0 else None for i in range(n)
            ],  # Some None values
            "name": [f"SpanName{i}" for i in range(n)],
            "status_code": [random.choice(["OK", "ERROR"]) for _ in range(n)],
            "status_message": [f"Message{i}" for i in range(n)],
            "events": pd.Series(
                mock_list_dicts(
                    n,
                    ["name", "timestamp", "attributes"],
                    list_length=random.randint(1, 3),
                )
            ),
            "span_kind": [random.choice(["INTERNAL", "CLIENT", "SERVER"]) for _ in range(n)],
            # Attributes
            "attributes.exception_type": [
                None if i % 3 == 0 else f"ExceptionType{i}" for i in range(n)
            ],
            "attributes.exception_message": [
                None if i % 3 == 0 else f"ExceptionMessage{i}" for i in range(n)
            ],
            "attributes.exception_escaped": [bool(i % 2) for i in range(n)],
            "attributes.exception_stacktrace": [
                None if i % 3 == 0 else f"StackTrace{i}" for i in range(n)
            ],
            "attributes.input_value": [f"InputValue{i}" for i in range(n)],
            "attributes.input_mime_type": [f"MIMEType{i}" for i in range(n)],
            "attributes.output_value": [f"OutputValue{i}" for i in range(n)],
            "attributes.output_mime_type": [f"MIMEType{i}" for i in range(n)],
            "attributes.embedding_model_name": [f"ModelName{i}" for i in range(n)],
            "attributes.embedding_embeddings": mock_list_dicts(
                n, ["embedding_vector", "embedding_text"]
            ),
            "attributes.llm_model_name": [f"LLMModelName{i}" for i in range(n)],
            "attributes.llm_input_messages": mock_list_dicts(
                n,
                ["name", "role", "content", "tool_calls"],
                list_length=random.randint(1, 3),
            ),
            "attributes.llm_output_messages": mock_list_dicts(
                n,
                ["name", "role", "content", "tool_calls"],
                list_length=random.randint(1, 3),
            ),
            "attributes.llm_invocation_parameters": mock_dict(n, ["param1", "param2"]),
            "attributes.llm_prompt_template_template": [f"Template{i}" for i in range(n)],
            "attributes.llm_prompt_template_variables": mock_dict(n, ["var1", "var2"]),
            "attributes.llm_prompt_template_version": [f"Version{i}" for i in range(n)],
            "attributes.llm_prompt_token_count": np.random.randint(1, 1000, n),
            "attributes.llm_completion_token_count": np.random.randint(1, 1000, n),
            "attributes.llm_total_token_count": np.random.randint(1, 2000, n),
            "attributes.tool_name": [f"ToolName{i}" for i in range(n)],
            "attributes.tool_description": [f"ToolDescription{i}" for i in range(n)],
            "attributes.tool_parameters": mock_dict(n, ["param1", "param2"]),
            "attributes.retrieval_documents": mock_list_dicts(
                n,
                [
                    "document_id",
                    "document_score",
                    "document_content",
                    "document_metadata",
                ],
                list_length=random.randint(1, 3),
            ),
            "attributes.reranker_input_documents": mock_list_dicts(
                n,
                [
                    "document_id",
                    "document_score",
                    "document_content",
                    "document_metadata",
                ],
                list_length=random.randint(1, 3),
            ),
            "attributes.reranker_output_documents": mock_list_dicts(
                n,
                [
                    "document_id",
                    "document_score",
                    "document_content",
                    "document_metadata",
                ],
                list_length=random.randint(1, 3),
            ),
            "attributes.reranker_query": [f"Query{i}" for i in range(n)],
            "attributes.reranker_model_name": [f"RerankerModelName{i}" for i in range(n)],
            "attributes.reranker_top_k": np.random.randint(1, 10, n),
        }
    )
    start_times, end_times = random_timestamps(n)
    df["start_time"] = pd.Series(start_times)
    df["end_time"] = pd.Series(end_times)
    return df


def generate_mock_eval_data(input_df: pd.DataFrame) -> pd.DataFrame:
    # Prepare data containers for each column
    span_ids = input_df["context.span_id"].values
    labels = [f"Label_{i}" for i in range(len(span_ids))]
    scores = np.random.rand(len(span_ids))
    explanations = [f"Explanation for outcome {label}" for label in labels]

    # Create the new DataFrame
    eval_df = pd.DataFrame(
        {
            "context.span_id": span_ids,
            "eval.test_eval.label": labels,
            "eval.test_eval.score": scores,
            "eval.test_eval.explanation": explanations,
        }
    )

    return eval_df


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors():
    try:
        log_spans(
            df=generate_mock_data(10),
        )
    except Exception:
        pytest.fail("Unexpected error")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def test_zero_errors_with_evals():
    df = generate_mock_data(10)
    evals_df = generate_mock_eval_data(df)
    try:
        log_spans(
            df=generate_mock_data(10),
            evals_df=evals_df,
        )
    except Exception:
        pytest.fail("Unexpected error")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires python>=3.8")
def log_spans(df: pd.DataFrame, evals_df: Optional[pd.DataFrame] = None) -> Response:
    client = NoSendClient("apikey", "spaceKey")
    response = client.log_spans(
        dataframe=df,
        evals_dataframe=evals_df,
        model_id="model-id",
        model_version="1.0",
    )
    return response


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
