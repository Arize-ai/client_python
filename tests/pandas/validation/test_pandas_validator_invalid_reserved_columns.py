from collections import ChainMap

import pandas as pd
import pytest

from arize.pandas.logger import Schema
from arize.pandas.validation.errors import ReservedColumns
from arize.pandas.validation.validator import Validator
from arize.utils.constants import (
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
)
from arize.utils.types import (
    EmbeddingColumnNames,
    Environments,
    LLMRunMetadataColumnNames,
    ModelTypes,
)


def test_valid_no_reserved_columns():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(feat_shap=[0]),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    tag_column_names=[
                        "tag_1",
                    ],
                    prompt_column_names=EmbeddingColumnNames(
                        vector_column_name="prompt_vector",
                        data_column_name="prompt",
                    ),
                    response_column_names=EmbeddingColumnNames(
                        vector_column_name="response_vector",
                        data_column_name="response",
                    ),
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 0


def test_valid_reserved_columns_in_correct_schema_fields():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(feat_shap=[0]),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    llm_run_metadata_column_names=LLMRunMetadataColumnNames(
                        total_token_count_column_name="total_token_count",
                        prompt_token_count_column_name="prompt_token_count",
                        response_token_count_column_name="response_token_count",
                        response_latency_ms_column_name="response_latency_ms",
                    ),
                    prompt_column_names=EmbeddingColumnNames(
                        vector_column_name="prompt_vector",
                        data_column_name="prompt",
                    ),
                    response_column_names=EmbeddingColumnNames(
                        vector_column_name="response_vector",
                        data_column_name="response",
                    ),
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 0


def test_valid_reserved_columns_non_generative_llm_model():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(feat_shap=[0]),
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_label_column_name="prediction_label",
                    tag_column_names=[
                        LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
                        LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
                        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
                        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
                    ],
                ),
                "model_type": ModelTypes.SCORE_CATEGORICAL,
            },
            kwargs,
        ),
    )
    assert len(errors) == 0


def test_invalid_reserved_columns_in_wrong_schema_fields():
    errors = Validator.validate_params(
        **ChainMap(
            {
                "dataframe": kwargs["dataframe"].assign(feat_shap=[0]),
                "schema": Schema(
                    prediction_id_column_name=LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
                    prediction_label_column_name=LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
                    feature_column_names=[
                        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                    ],
                    tag_column_names=[
                        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
                    ],
                    prompt_column_names=EmbeddingColumnNames(
                        vector_column_name="prompt_vector",
                        data_column_name="prompt",
                    ),
                    response_column_names=EmbeddingColumnNames(
                        vector_column_name="response_vector",
                        data_column_name="response",
                    ),
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert isinstance(errors[0], ReservedColumns)
    assert len(errors[0].reserved_columns) == 4


kwargs = {
    "model_id": "fraud",
    "model_version": "v1.0",
    "model_type": ModelTypes.GENERATIVE_LLM,
    "environment": Environments.PRODUCTION,
    "dataframe": pd.DataFrame(
        {
            "prediction_id": pd.Series(["0"]),
            "prediction_label": pd.Series(["fraud"]),
            "prediction_score": pd.Series([1]),
            "actual_label": pd.Series(["not fraud"]),
            "actual_score": pd.Series([0]),
            "tag_1": pd.Series(["tag_1_value"]),
            "prompt": pd.Series(["this is a prompt"]),
            "prompt_vector": pd.Series([[0.1, 0.2]]),
            "response": pd.Series(["this is a response"]),
            "response_vector": pd.Series([[0.5, 1.0]]),
            LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME: pd.Series([16]),
            LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME: pd.Series([10]),
            LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME: pd.Series([6]),
            LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME: pd.Series([1000]),
        }
    ),
    "schema": Schema(
        prediction_id_column_name="prediction_id",
        prediction_label_column_name="prediction_label",
        prompt_column_names=EmbeddingColumnNames(
            vector_column_name="prompt_vector",
            data_column_name="prompt",
        ),
        response_column_names=EmbeddingColumnNames(
            vector_column_name="response_vector",
            data_column_name="response",
        ),
    ),
}

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
