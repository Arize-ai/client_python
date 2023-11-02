from collections import ChainMap

import arize.pandas.validation.errors as err
import pandas as pd
import pytest
from arize.pandas.validation.validator import Validator
from arize.utils.types import (
    CorpusSchema,
    EmbeddingColumnNames,
    Environments,
    LLMConfigColumnNames,
    Schema,
)


def get_standard_kwargs():
    return {
        "dataframe": pd.DataFrame(
            {
                "prediction_id": pd.Series(["0", "1", "2", "3", "4"]),
                "prediction_score": [1, -2, 6, 7, -4],
            }
        ),
        "model_id": "fraud",
        "environment": Environments.PRODUCTION,
        "schema": Schema(
            prediction_id_column_name="prediction_id",
            prediction_score_column_name="prediction_score",
            embedding_feature_column_names={
                "image_embedding": EmbeddingColumnNames(
                    vector_column_name="image_vector",
                    link_to_data_column_name="image_link",
                ),
            },
            prompt_column_names=EmbeddingColumnNames(
                vector_column_name="prompt_vector",
                data_column_name="prompt",
            ),
            response_column_names=EmbeddingColumnNames(
                vector_column_name="response_vector",
                data_column_name="response",
            ),
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
    assert isinstance(errors[0], err.InvalidIndex)


def test_field_type_embedding_features_column_names():
    kwargs = get_standard_kwargs()
    embedding_feature_column_correct = kwargs["schema"].embedding_feature_column_names
    embedding_feature_column_incorrect = [
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
                    embedding_feature_column_names=embedding_feature_column_correct,  # type: ignore
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 0

    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    embedding_feature_column_names=embedding_feature_column_incorrect,  # type: ignore
                ),  # type: ignore
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidFieldTypeEmbeddingFeatures)


def test_field_type_prompt_response_column_names():
    # This contains a dataframe with EmbeddingColumNames prompt & response
    kwargs = get_standard_kwargs()
    schema = kwargs["schema"]

    errors = Validator.validate_required_checks(
        **ChainMap(
            kwargs,
        ),
    )
    assert len(errors) == 0

    schema = schema.replace(
        prompt_column_names="prompt_column",
        response_column_names="response_column",
    )
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": schema,
            },
            kwargs,
        ),
    )
    assert len(errors) == 0

    # This gives string objects
    schema = schema.replace(
        prompt_column_names=2,
        response_column_names=2,
    )
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": schema,
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 2
    for e in errors:
        assert isinstance(e, err.InvalidFieldTypePromptResponse)


def test_field_type_prompt_template_and_llm_config_column_names():
    kwargs = get_standard_kwargs()
    # This gives string objects
    prompt_template_column_names_incorrect = "prompt_templates"
    llm_config_column_names_incorrect = "llm_config"

    errors = Validator.validate_required_checks(
        **ChainMap(
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
                    prompt_template_column_names=prompt_template_column_names_incorrect,  # type: ignore
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidFieldTypePromptTemplates)

    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    llm_config_column_names=llm_config_column_names_incorrect,  # type:ignore
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidFieldTypeLlmConfig)


def test_type_llm_params():
    kwargs = get_standard_kwargs()
    dataframe_correct = kwargs["dataframe"]
    dataframe_correct["llm_params"] = [
        {"temperature": i / 4, "presence_penalty": i / 3, "stop": [".", "?", "!"]} for i in range(5)
    ]
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "dataframe": dataframe_correct,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    llm_config_column_names=LLMConfigColumnNames(params_column_name="llm_params"),
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 0

    dataframe_incorrect = dataframe_correct
    dataframe_incorrect["llm_params"] = [i for i in range(5)]
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "dataframe": dataframe_incorrect,
                "schema": Schema(
                    prediction_id_column_name="prediction_id",
                    prediction_score_column_name="prediction_score",
                    llm_config_column_names=LLMConfigColumnNames(params_column_name="llm_params"),
                ),
            },
            kwargs,
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidTypeColumns)


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
        ),  # type: ignore
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidFieldTypeConversion)
    assert "model_id" in errors[0].error_message()
    assert "model_version" in errors[0].error_message()
    assert "batch_id" in errors[0].error_message()


def test_invalid_schema_type():
    kwargs = get_standard_kwargs()
    # Corpus schema with Corpus environment should pass
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": CorpusSchema(
                    document_id_column_name="document_id",
                    document_version_column_name="document_version",
                    document_text_embedding_column_names=EmbeddingColumnNames(
                        vector_column_name="document_vector",
                        data_column_name="document_data",
                    ),
                ),
                "environment": Environments.CORPUS,
            },
            kwargs,
        ),
    )
    assert len(errors) == 0

    # Corpus schema with non Corpus environment should not pass
    errors = Validator.validate_required_checks(
        **ChainMap(
            {
                "schema": CorpusSchema(
                    document_id_column_name="document_id",
                    document_version_column_name="document_version",
                    document_text_embedding_column_names=EmbeddingColumnNames(
                        vector_column_name="document_vector",
                        data_column_name="document_data",
                    ),
                ),
                "environment": Environments.PRODUCTION,
            },
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidSchemaType)

    # non Corpus schema with Corpus environment should not pass
    errors = Validator.validate_required_checks(
        **ChainMap(
            {"environment": Environments.CORPUS},
            kwargs,
        ),
    )
    assert len(errors) == 1
    assert isinstance(errors[0], err.InvalidSchemaType)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
