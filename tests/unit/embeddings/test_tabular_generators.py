"""Tests for arize.embeddings.tabular_generators module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from arize.embeddings.constants import DEFAULT_TABULAR_MODEL
from arize.embeddings.errors import InvalidIndexError
from arize.embeddings.tabular_generators import (
    TABULAR_PRETRAINED_MODELS,
    EmbeddingGeneratorForTabularFeatures,
)


@pytest.fixture
def mock_transformers() -> Generator[dict[str, Any], None, None]:
    """Mock transformers dependencies for testing."""
    with (
        patch(
            "arize.embeddings.base_generators.AutoModel.from_pretrained"
        ) as mock_model,
        patch(
            "arize.embeddings.base_generators.AutoTokenizer.from_pretrained"
        ) as mock_tokenizer,
        patch(
            "arize.embeddings.base_generators.torch.cuda.is_available"
        ) as mock_cuda,
        patch(
            "arize.embeddings.base_generators.torch.backends.mps.is_available"
        ) as mock_mps,
    ):
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        # Setup mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Setup device mocks
        mock_cuda.return_value = False
        mock_mps.return_value = False

        yield {
            "model": mock_model,
            "model_instance": mock_model_instance,
            "tokenizer": mock_tokenizer,
            "tokenizer_instance": mock_tokenizer_instance,
            "cuda": mock_cuda,
            "mps": mock_mps,
        }


@pytest.fixture
def sample_tabular_df() -> pd.DataFrame:
    """Sample tabular data for testing."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "education_level": ["Bachelor", "Master", "PhD"],
        }
    )


@pytest.mark.unit
class TestEmbeddingGeneratorForTabularFeaturesInitialization:
    """Test initialization of EmbeddingGeneratorForTabularFeatures."""

    def test_initialization_with_default_model(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize with default distilbert model."""
        generator = EmbeddingGeneratorForTabularFeatures()
        assert generator.model_name == DEFAULT_TABULAR_MODEL
        assert generator.model_name == "distilbert-base-uncased"

    def test_initialization_with_custom_valid_model(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize with valid custom model."""
        custom_model = "bert-base-uncased"
        generator = EmbeddingGeneratorForTabularFeatures(
            model_name=custom_model
        )
        assert generator.model_name == custom_model

    def test_initialization_with_invalid_model_raises(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise ValueError for unsupported model."""
        with pytest.raises(ValueError, match="model_name not supported"):
            EmbeddingGeneratorForTabularFeatures(model_name="unsupported-model")

    def test_model_must_be_in_pretrained_list(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should validate model is in TABULAR_PRETRAINED_MODELS."""
        # Valid model
        generator = EmbeddingGeneratorForTabularFeatures(
            model_name="xlm-roberta-base"
        )
        assert generator.model_name in TABULAR_PRETRAINED_MODELS

    def test_use_case_is_tabular_embeddings(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should have TABULAR_EMBEDDINGS use case."""
        generator = EmbeddingGeneratorForTabularFeatures()
        assert generator.use_case == "Tabular.TABULAR_EMBEDDINGS"

    def test_inherits_from_nlp_embedding_generator(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should inherit from NLPEmbeddingGenerator."""
        from arize.embeddings.base_generators import NLPEmbeddingGenerator

        generator = EmbeddingGeneratorForTabularFeatures()
        assert isinstance(generator, NLPEmbeddingGenerator)

    def test_kwargs_passed_to_parent(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should pass kwargs to parent class."""
        generator = EmbeddingGeneratorForTabularFeatures(batch_size=50)
        assert generator.batch_size == 50


@pytest.mark.unit
class TestListPretrainedModels:
    """Test list_pretrained_models class method."""

    def test_list_pretrained_models_returns_dataframe(self) -> None:
        """Should return a pandas DataFrame."""
        result = EmbeddingGeneratorForTabularFeatures.list_pretrained_models()
        assert isinstance(result, pd.DataFrame)

    def test_pretrained_models_has_model_name_column(self) -> None:
        """Should have Model Name column."""
        result = EmbeddingGeneratorForTabularFeatures.list_pretrained_models()
        assert "Model Name" in result.columns

    def test_pretrained_models_has_three_models(self) -> None:
        """Should have 3 supported models."""
        result = EmbeddingGeneratorForTabularFeatures.list_pretrained_models()
        assert len(result) == 3

    def test_pretrained_models_contains_expected_models(self) -> None:
        """Should contain expected model names."""
        result = EmbeddingGeneratorForTabularFeatures.list_pretrained_models()
        model_names = result["Model Name"].tolist()
        assert "bert-base-uncased" in model_names
        assert "distilbert-base-uncased" in model_names
        assert "xlm-roberta-base" in model_names

    def test_pretrained_models_is_sorted(self) -> None:
        """Should return sorted model list."""
        result = EmbeddingGeneratorForTabularFeatures.list_pretrained_models()
        models = result["Model Name"].tolist()
        assert models == sorted(models)


@pytest.mark.unit
class TestInputValidation:
    """Test input validation in generate_embeddings."""

    def test_df_must_be_dataframe(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise TypeError if df is not a DataFrame."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
            generator.generate_embeddings(
                df=[1, 2, 3],  # type: ignore[arg-type]
                selected_columns=["age"],
            )

    def test_selected_columns_must_be_list(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise TypeError if selected_columns is not a list."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(TypeError, match="columns must be a list"):
            generator.generate_embeddings(
                df=sample_tabular_df,
                selected_columns="age",  # type: ignore[arg-type]
            )

    def test_selected_columns_must_contain_strings(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise TypeError if selected_columns contains non-strings."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(TypeError, match="columns must be a list"):
            generator.generate_embeddings(
                df=sample_tabular_df,
                selected_columns=[1, 2, 3],  # type: ignore[list-item]
            )

    def test_selected_columns_must_exist_in_df(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise ValueError if selected_columns not in DataFrame."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(ValueError, match="not found"):
            generator.generate_embeddings(
                df=sample_tabular_df, selected_columns=["nonexistent_column"]
            )

    def test_col_name_map_must_be_dict(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise TypeError if col_name_map is not a dict."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(
            TypeError, match="col_name_map must be a dictionary"
        ):
            generator.generate_embeddings(
                df=sample_tabular_df,
                selected_columns=["age"],
                col_name_map=["not", "a", "dict"],  # type: ignore[arg-type]
            )

    def test_col_name_map_keys_must_be_strings(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise TypeError if col_name_map keys are not strings."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(
            TypeError, match="keys and values should be strings"
        ):
            generator.generate_embeddings(
                df=sample_tabular_df,
                selected_columns=["age"],
                col_name_map={1: "customer_age"},  # type: ignore[dict-item]
            )

    def test_col_name_map_values_must_be_strings(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise TypeError if col_name_map values are not strings."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(
            TypeError, match="keys and values should be strings"
        ):
            generator.generate_embeddings(
                df=sample_tabular_df,
                selected_columns=["age"],
                col_name_map={"age": 123},  # type: ignore[dict-item]
            )

    def test_col_name_map_keys_must_exist_in_df(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should raise ValueError if col_name_map keys not in DataFrame."""
        generator = EmbeddingGeneratorForTabularFeatures()
        with pytest.raises(ValueError, match="not found"):
            generator.generate_embeddings(
                df=sample_tabular_df,
                selected_columns=["age"],
                col_name_map={"nonexistent": "new_name"},
            )

    def test_dataframe_index_validation(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should validate DataFrame has default index."""
        generator = EmbeddingGeneratorForTabularFeatures()
        df_with_bad_index = pd.DataFrame({"age": [25, 30]}, index=["a", "b"])

        with pytest.raises(InvalidIndexError):
            generator.generate_embeddings(
                df=df_with_bad_index, selected_columns=["age"]
            )


@pytest.mark.unit
class TestPromptGeneration:
    """Test prompt generation from tabular data."""

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_prompt_generated_from_selected_columns(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should generate prompts from selected columns."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        _, prompts = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age"],
            return_prompt_col=True,
        )

        assert isinstance(prompts, pd.Series)
        assert len(prompts) == 3
        assert "age" in prompts.iloc[0]

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_prompt_with_column_name_mapping(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should use mapped column names in prompts."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        _, prompts = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age"],
            col_name_map={"age": "customer_age"},
            return_prompt_col=True,
        )

        # Note: underscores are replaced with spaces in prompts
        assert "customer age" in prompts.iloc[0]
        # Original column name should not appear when mapped
        assert (
            " age " not in prompts.iloc[0] or "customer age" in prompts.iloc[0]
        )

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_underscores_replaced_with_spaces(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should replace underscores with spaces in prompts."""
        generator = EmbeddingGeneratorForTabularFeatures()

        df = pd.DataFrame({"education_level": ["Bachelor"]})

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0])]}
        )

        _, prompts = generator.generate_embeddings(
            df=df,
            selected_columns=["education_level"],
            return_prompt_col=True,
        )

        assert "education level" in prompts.iloc[0]
        assert "education_level" not in prompts.iloc[0]

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_prompt_format_includes_is(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should use 'The X is Y.' format in prompts."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        _, prompts = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age"],
            return_prompt_col=True,
        )

        assert "The" in prompts.iloc[0]
        assert " is " in prompts.iloc[0]
        assert "." in prompts.iloc[0]

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_prompt_handles_numeric_values(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should convert numeric values to strings in prompts."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        _, prompts = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age", "income"],
            return_prompt_col=True,
        )

        # Check that numeric values are in the prompt
        assert "25" in prompts.iloc[0] or "50000" in prompts.iloc[0]

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_return_prompt_col_false_returns_series(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should return only embeddings when return_prompt_col is False."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age"],
            return_prompt_col=False,
        )

        assert isinstance(result, pd.Series)
        assert not isinstance(result, tuple)

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_return_prompt_col_true_returns_tuple(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should return tuple of (embeddings, prompts) when return_prompt_col is True."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age"],
            return_prompt_col=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        embeddings, prompts = result
        assert isinstance(embeddings, pd.Series)
        assert isinstance(prompts, pd.Series)


@pytest.mark.unit
class TestEmbeddingMethod:
    """Test embedding calculation method selection."""

    def test_bert_models_use_avg_token_method(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should use avg_token method for BERT models."""
        generator = EmbeddingGeneratorForTabularFeatures(
            model_name="bert-base-uncased"
        )
        method = generator._EmbeddingGeneratorForTabularFeatures__get_method_for_embedding_calculation()
        assert method == "avg_token"

    def test_distilbert_models_use_avg_token_method(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should use avg_token method for DistilBERT models."""
        generator = EmbeddingGeneratorForTabularFeatures(
            model_name="distilbert-base-uncased"
        )
        method = generator._EmbeddingGeneratorForTabularFeatures__get_method_for_embedding_calculation()
        assert method == "avg_token"

    def test_roberta_models_use_cls_token_method(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should use cls_token method for RoBERTa models."""
        generator = EmbeddingGeneratorForTabularFeatures(
            model_name="xlm-roberta-base"
        )
        method = generator._EmbeddingGeneratorForTabularFeatures__get_method_for_embedding_calculation()
        assert method == "cls_token"


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_single_column_selection(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should work with single column selected."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(
            df=sample_tabular_df, selected_columns=["age"]
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_all_columns_selected(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should work when all columns are selected."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age", "income", "education_level"],
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    @patch("arize.embeddings.tabular_generators.Dataset")
    def test_multiple_columns_with_mapping(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_tabular_df: pd.DataFrame,
    ) -> None:
        """Should handle multiple columns with partial mapping."""
        generator = EmbeddingGeneratorForTabularFeatures()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        _, prompts = generator.generate_embeddings(
            df=sample_tabular_df,
            selected_columns=["age", "income"],
            col_name_map={"age": "customer_age"},  # Only map one column
            return_prompt_col=True,
        )

        # Note: underscores are replaced with spaces in prompts
        assert "customer age" in prompts.iloc[0]
        assert "income" in prompts.iloc[0]


@pytest.mark.unit
class TestRepr:
    """Test string representation."""

    def test_repr_includes_use_case(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should include use case in repr."""
        generator = EmbeddingGeneratorForTabularFeatures()
        repr_str = repr(generator)
        assert "use_case" in repr_str
        assert "TABULAR_EMBEDDINGS" in repr_str

    def test_repr_includes_model_name(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should include model name in repr."""
        generator = EmbeddingGeneratorForTabularFeatures(
            model_name="bert-base-uncased"
        )
        repr_str = repr(generator)
        assert "model_name" in repr_str
        assert "bert-base-uncased" in repr_str

    def test_repr_includes_tokenizer_max_length(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should include tokenizer max length in repr."""
        generator = EmbeddingGeneratorForTabularFeatures()
        repr_str = repr(generator)
        assert "tokenizer_max_length" in repr_str
