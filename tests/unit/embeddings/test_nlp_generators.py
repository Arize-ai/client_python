"""Tests for arize.embeddings.nlp_generators module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from arize.embeddings.constants import (
    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
    DEFAULT_NLP_SUMMARIZATION_MODEL,
)
from arize.embeddings.errors import InvalidIndexError
from arize.embeddings.nlp_generators import (
    EmbeddingGeneratorForNLPSequenceClassification,
    EmbeddingGeneratorForNLPSummarization,
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
def sample_text_series() -> pd.Series:
    """Sample text data for testing."""
    return pd.Series(
        [
            "This is a test sentence.",
            "Another test sentence.",
            "Third test sentence.",
        ]
    )


@pytest.fixture
def sample_class_labels() -> pd.Series:
    """Sample class labels for testing."""
    return pd.Series(["positive", "negative", "neutral"])


@pytest.mark.unit
class TestEmbeddingGeneratorForNLPSequenceClassification:
    """Test EmbeddingGeneratorForNLPSequenceClassification class."""

    def test_initialization_with_default_model(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize with default distilbert model."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.model_name == DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL
        assert generator.model_name == "distilbert-base-uncased"

    def test_initialization_with_custom_model(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize with custom model name."""
        custom_model = "bert-base-cased"
        generator = EmbeddingGeneratorForNLPSequenceClassification(
            model_name=custom_model
        )
        assert generator.model_name == custom_model

    def test_use_case_is_sequence_classification(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should have SEQUENCE_CLASSIFICATION use case."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.use_case == "NLP.SEQUENCE_CLASSIFICATION"

    def test_inherits_from_nlp_embedding_generator(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should inherit from NLPEmbeddingGenerator."""
        from arize.embeddings.base_generators import NLPEmbeddingGenerator

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert isinstance(generator, NLPEmbeddingGenerator)

    def test_tokenizer_initialized(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize tokenizer during construction."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.tokenizer is not None

    def test_kwargs_passed_to_parent(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should pass kwargs to parent class."""
        generator = EmbeddingGeneratorForNLPSequenceClassification(
            batch_size=50
        )
        assert generator.batch_size == 50

    def test_text_col_must_be_series(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise TypeError if text_col is not a pandas Series."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()
        with pytest.raises(TypeError, match="text_col must be a pandas Series"):
            generator.generate_embeddings(text_col=["not", "a", "series"])  # type: ignore[arg-type]

    def test_class_label_col_must_be_series(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
    ) -> None:
        """Should raise TypeError if class_label_col is not a pandas Series."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()
        with pytest.raises(
            TypeError, match="class_label_col must be a pandas Series"
        ):
            generator.generate_embeddings(
                text_col=sample_text_series,
                class_label_col=["not", "a", "series"],  # type: ignore[arg-type]
            )

    @patch("arize.embeddings.nlp_generators.Dataset")
    def test_generate_embeddings_with_text_only(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
    ) -> None:
        """Should generate embeddings with text_col only."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(text_col=sample_text_series)

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    @patch("arize.embeddings.nlp_generators.Dataset")
    def test_generate_embeddings_with_class_labels(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
        sample_class_labels: pd.Series,
    ) -> None:
        """Should append class labels to text when class_label_col provided."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(
            text_col=sample_text_series, class_label_col=sample_class_labels
        )

        assert isinstance(result, pd.Series)
        # Verify Dataset.from_dict was called
        mock_dataset.from_dict.assert_called_once()

    @patch("arize.embeddings.nlp_generators.Dataset")
    def test_class_label_prepended_to_text(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
        sample_class_labels: pd.Series,
    ) -> None:
        """Should prepend 'The classification label is <label>' to text."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        generator.generate_embeddings(
            text_col=sample_text_series, class_label_col=sample_class_labels
        )

        # Check that Dataset.from_dict was called with modified text
        call_args = mock_dataset.from_dict.call_args
        text_values = call_args[0][0]["text"]
        # Check that the first text contains the classification label
        assert "The classification label is" in text_values.iloc[0]

    def test_invalid_index_raises_error(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise InvalidIndexError for non-default index."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()
        text_with_bad_index = pd.Series(["text1", "text2"], index=["a", "b"])

        with pytest.raises(InvalidIndexError):
            generator.generate_embeddings(text_col=text_with_bad_index)

    @patch("arize.embeddings.nlp_generators.Dataset")
    def test_returns_series_with_embeddings(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
    ) -> None:
        """Should return pd.Series containing embedding vectors."""
        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0, 3.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(text_col=sample_text_series)

        assert isinstance(result, pd.Series)
        assert result.name == "embedding_vector"


@pytest.mark.unit
class TestEmbeddingGeneratorForNLPSummarization:
    """Test EmbeddingGeneratorForNLPSummarization class."""

    def test_initialization_with_default_model(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize with default distilbert model."""
        generator = EmbeddingGeneratorForNLPSummarization()
        assert generator.model_name == DEFAULT_NLP_SUMMARIZATION_MODEL
        assert generator.model_name == "distilbert-base-uncased"

    def test_initialization_with_custom_model(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize with custom model name."""
        custom_model = "bert-large-uncased"
        generator = EmbeddingGeneratorForNLPSummarization(
            model_name=custom_model
        )
        assert generator.model_name == custom_model

    def test_use_case_is_summarization(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should have SUMMARIZATION use case."""
        generator = EmbeddingGeneratorForNLPSummarization()
        assert generator.use_case == "NLP.SUMMARIZATION"

    def test_inherits_from_nlp_embedding_generator(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should inherit from NLPEmbeddingGenerator."""
        from arize.embeddings.base_generators import NLPEmbeddingGenerator

        generator = EmbeddingGeneratorForNLPSummarization()
        assert isinstance(generator, NLPEmbeddingGenerator)

    def test_tokenizer_initialized(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should initialize tokenizer during construction."""
        generator = EmbeddingGeneratorForNLPSummarization()
        assert generator.tokenizer is not None

    def test_kwargs_passed_to_parent(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should pass kwargs to parent class."""
        generator = EmbeddingGeneratorForNLPSummarization(batch_size=75)
        assert generator.batch_size == 75

    def test_text_col_must_be_series(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise TypeError if text_col is not a pandas Series."""
        generator = EmbeddingGeneratorForNLPSummarization()
        with pytest.raises(TypeError, match="text_col must be a pandas Series"):
            generator.generate_embeddings(text_col=["not", "a", "series"])  # type: ignore[arg-type]

    @patch("arize.embeddings.nlp_generators.Dataset")
    def test_generate_embeddings_with_text(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
    ) -> None:
        """Should generate embeddings from text_col."""
        generator = EmbeddingGeneratorForNLPSummarization()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([1.0, 2.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(text_col=sample_text_series)

        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_invalid_index_raises_error(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise InvalidIndexError for non-default index."""
        generator = EmbeddingGeneratorForNLPSummarization()
        text_with_bad_index = pd.Series(["text1", "text2"], index=["a", "b"])

        with pytest.raises(InvalidIndexError):
            generator.generate_embeddings(text_col=text_with_bad_index)

    @patch("arize.embeddings.nlp_generators.Dataset")
    def test_returns_series_with_embeddings(
        self,
        mock_dataset: MagicMock,
        mock_transformers: Generator[dict[str, Any], None, None],
        sample_text_series: pd.Series,
    ) -> None:
        """Should return pd.Series containing embedding vectors."""
        generator = EmbeddingGeneratorForNLPSummarization()

        # Mock Dataset behavior
        mock_ds_instance = MagicMock()
        mock_dataset.from_dict.return_value = mock_ds_instance
        mock_ds_instance.map.return_value = mock_ds_instance
        mock_ds_instance.to_pandas.return_value = pd.DataFrame(
            {"embedding_vector": [np.array([4.0, 5.0, 6.0]) for _ in range(3)]}
        )

        result = generator.generate_embeddings(text_col=sample_text_series)

        assert isinstance(result, pd.Series)
        assert result.name == "embedding_vector"


@pytest.mark.unit
class TestNLPGeneratorsComparison:
    """Test differences between NLP generator classes."""

    def test_same_default_models(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should have same default model (distilbert-base-uncased)."""
        seq_gen = EmbeddingGeneratorForNLPSequenceClassification()
        sum_gen = EmbeddingGeneratorForNLPSummarization()

        assert seq_gen.model_name == sum_gen.model_name

    def test_different_use_cases(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should have different use cases assigned."""
        seq_gen = EmbeddingGeneratorForNLPSequenceClassification()
        sum_gen = EmbeddingGeneratorForNLPSummarization()

        assert seq_gen.use_case != sum_gen.use_case
        assert seq_gen.use_case == "NLP.SEQUENCE_CLASSIFICATION"
        assert sum_gen.use_case == "NLP.SUMMARIZATION"

    def test_sequence_classification_has_class_label_parameter(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should have class_label_col parameter in sequence classification."""
        from inspect import signature

        sig = signature(
            EmbeddingGeneratorForNLPSequenceClassification.generate_embeddings
        )
        assert "class_label_col" in sig.parameters

    def test_summarization_has_no_class_label_parameter(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should not have class_label_col parameter in summarization."""
        from inspect import signature

        sig = signature(
            EmbeddingGeneratorForNLPSummarization.generate_embeddings
        )
        assert "class_label_col" not in sig.parameters
