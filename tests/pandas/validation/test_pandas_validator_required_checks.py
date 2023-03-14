from collections import ChainMap

import pandas as pd
import pytest
from arize.pandas.logger import Schema
from arize.pandas.validation.errors import (
    InvalidFieldTypeConversion,
    InvalidFieldTypeEmbeddingFeatures,
    InvalidIndex,
)
from arize.pandas.validation.validator import Validator
from arize.utils.types import EmbeddingColumnNames


def get_standard_kwargs():
    return {
        "dataframe": pd.DataFrame(
            {
                "prediction_id": pd.Series(["0", "1", "2", "3", "4"]),
                "prediction_score": [1, -2, 6, 7, -4],
            }
        ),
        "model_id": "fraud",
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            prediction_score_column_name="prediction_score",
            embedding_feature_column_names={
                "image_embedding": EmbeddingColumnNames(
                    vector_column_name="image_vector",
                    link_to_data_column_name="image_link",
                ),
            },
        ),
        "model_version": "v1",
        "batch_id": "validation",
    }


def test_invalid_index():
    kwargs = get_standard_kwargs()
    df_correct = kwargs["dataframe"]
    df_incorrect = df_correct.loc[df_correct.prediction_score > 0]
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "dataframe": df_correct,
            },
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "dataframe": df_incorrect,
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert isinstance(errors[0], InvalidIndex)


def test_field_type_embedding_features_column_names():
    kwargs = get_standard_kwargs()
    embeddings_embedding_feature_column_correct = kwargs["schema"].embedding_feature_column_names
    embeddings_embedding_feature_column_incorrect = [
        EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="image_link",
        )
    ]
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    embedding_feature_column_names=embeddings_embedding_feature_column_correct,
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 0

    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    embedding_feature_column_names=embeddings_embedding_feature_column_incorrect,
                ),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert isinstance(errors[0], InvalidFieldTypeEmbeddingFeatures)


def test_field_convertible_to_str():
    class Dummy_class:
        def __str__(self):
            pass

    kwargs = get_standard_kwargs()
    errors = Validator.validate_required_checks(
        **ChainMap(
            kwargs,
        ),
    )
    assert len(errors) == 0
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "model_id": Dummy_class(),
                "model_version": Dummy_class(),
                "batch_id": Dummy_class(),
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert isinstance(errors[0], InvalidFieldTypeConversion)
    assert "model_id" in errors[0].error_message()
    assert "model_version" in errors[0].error_message()
    assert "batch_id" in errors[0].error_message()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
