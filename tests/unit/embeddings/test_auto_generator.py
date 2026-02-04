"""Tests for arize.embeddings.auto_generator module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from arize.embeddings.auto_generator import EmbeddingGenerator
from arize.embeddings.cv_generators import (
    EmbeddingGeneratorForCVImageClassification,
    EmbeddingGeneratorForCVObjectDetection,
)
from arize.embeddings.nlp_generators import (
    EmbeddingGeneratorForNLPSequenceClassification,
    EmbeddingGeneratorForNLPSummarization,
)
from arize.embeddings.tabular_generators import (
    EmbeddingGeneratorForTabularFeatures,
)
from arize.embeddings.usecases import UseCases

if TYPE_CHECKING:
    from collections.abc import Generator


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
            "arize.embeddings.base_generators.AutoImageProcessor.from_pretrained"
        ) as mock_processor,
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
        mock_tokenizer.return_value = MagicMock()

        # Setup mock processor
        mock_processor.return_value = MagicMock()

        # Setup device mocks
        mock_cuda.return_value = False
        mock_mps.return_value = False

        yield {
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "processor": mock_processor,
            "cuda": mock_cuda,
            "mps": mock_mps,
        }


@pytest.mark.unit
class TestEmbeddingGeneratorFactory:
    """Test EmbeddingGenerator factory class."""

    def test_direct_instantiation_raises_os_error(self) -> None:
        """Should raise OSError when trying to instantiate directly."""
        with pytest.raises(OSError, match="from_use_case"):
            EmbeddingGenerator()

    def test_error_message_mentions_class_name(self) -> None:
        """Should include class name in error message."""
        with pytest.raises(OSError, match="EmbeddingGenerator"):
            EmbeddingGenerator()

    def test_error_message_mentions_factory_method(self) -> None:
        """Should mention from_use_case factory method in error message."""
        with pytest.raises(OSError, match="from_use_case"):
            EmbeddingGenerator()


@pytest.mark.unit
class TestFromUseCase:
    """Test from_use_case factory method."""

    def test_from_use_case_nlp_sequence_classification(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should return NLP sequence classification generator."""
        generator = EmbeddingGenerator.from_use_case(
            UseCases.NLP.SEQUENCE_CLASSIFICATION
        )
        assert isinstance(
            generator, EmbeddingGeneratorForNLPSequenceClassification
        )

    def test_from_use_case_nlp_summarization(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should return NLP summarization generator."""
        generator = EmbeddingGenerator.from_use_case(UseCases.NLP.SUMMARIZATION)
        assert isinstance(generator, EmbeddingGeneratorForNLPSummarization)

    def test_from_use_case_cv_image_classification(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should return CV image classification generator."""
        generator = EmbeddingGenerator.from_use_case(
            UseCases.CV.IMAGE_CLASSIFICATION
        )
        assert isinstance(generator, EmbeddingGeneratorForCVImageClassification)

    def test_from_use_case_cv_object_detection(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should return CV object detection generator."""
        generator = EmbeddingGenerator.from_use_case(
            UseCases.CV.OBJECT_DETECTION
        )
        assert isinstance(generator, EmbeddingGeneratorForCVObjectDetection)

    def test_from_use_case_tabular(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should return tabular features generator."""
        generator = EmbeddingGenerator.from_use_case(
            UseCases.STRUCTURED.TABULAR_EMBEDDINGS
        )
        assert isinstance(generator, EmbeddingGeneratorForTabularFeatures)

    def test_invalid_use_case_raises_value_error(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise ValueError for invalid use case."""
        with pytest.raises(ValueError, match="Invalid use case"):
            EmbeddingGenerator.from_use_case("invalid_use_case")

    def test_kwargs_passed_to_nlp_generator(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should forward kwargs to NLP generator."""
        generator = EmbeddingGenerator.from_use_case(
            UseCases.NLP.SEQUENCE_CLASSIFICATION,
            model_name="custom-bert",
            batch_size=50,
        )
        assert generator.model_name == "custom-bert"
        assert generator.batch_size == 50

    def test_kwargs_passed_to_cv_generator(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should forward kwargs to CV generator."""
        generator = EmbeddingGenerator.from_use_case(
            UseCases.CV.IMAGE_CLASSIFICATION,
            model_name="custom-vit",
            batch_size=25,
        )
        assert generator.model_name == "custom-vit"
        assert generator.batch_size == 25

    @pytest.mark.parametrize(
        "use_case,expected_class",
        [
            (
                UseCases.NLP.SEQUENCE_CLASSIFICATION,
                EmbeddingGeneratorForNLPSequenceClassification,
            ),
            (UseCases.NLP.SUMMARIZATION, EmbeddingGeneratorForNLPSummarization),
            (
                UseCases.CV.IMAGE_CLASSIFICATION,
                EmbeddingGeneratorForCVImageClassification,
            ),
            (
                UseCases.CV.OBJECT_DETECTION,
                EmbeddingGeneratorForCVObjectDetection,
            ),
            (
                UseCases.STRUCTURED.TABULAR_EMBEDDINGS,
                EmbeddingGeneratorForTabularFeatures,
            ),
        ],
    )
    def test_use_case_routing(
        self,
        mock_transformers: Generator[dict[str, Any], None, None],
        use_case: object,
        expected_class: object,
    ) -> None:
        """Should route to correct generator class for each use case."""
        generator = EmbeddingGenerator.from_use_case(use_case)
        assert isinstance(generator, expected_class)


@pytest.mark.unit
class TestListDefaultModels:
    """Test list_default_models method."""

    def test_list_default_models_returns_dataframe(self) -> None:
        """Should return a pandas DataFrame."""
        result = EmbeddingGenerator.list_default_models()
        assert isinstance(result, pd.DataFrame)

    def test_default_models_has_required_columns(self) -> None:
        """Should have Area, Usecase, and Model Name columns."""
        result = EmbeddingGenerator.list_default_models()
        assert "Area" in result.columns
        assert "Usecase" in result.columns
        assert "Model Name" in result.columns

    def test_default_models_has_five_rows(self) -> None:
        """Should have one row per use case (5 total)."""
        result = EmbeddingGenerator.list_default_models()
        assert len(result) == 5

    def test_default_models_contains_nlp_use_cases(self) -> None:
        """Should include NLP use cases."""
        result = EmbeddingGenerator.list_default_models()
        assert "SEQUENCE_CLASSIFICATION" in result["Usecase"].values
        assert "SUMMARIZATION" in result["Usecase"].values

    def test_default_models_contains_cv_use_cases(self) -> None:
        """Should include CV use cases."""
        result = EmbeddingGenerator.list_default_models()
        assert "IMAGE_CLASSIFICATION" in result["Usecase"].values
        assert "OBJECT_DETECTION" in result["Usecase"].values

    def test_default_models_contains_tabular_use_case(self) -> None:
        """Should include tabular use case."""
        result = EmbeddingGenerator.list_default_models()
        assert "TABULAR_EMBEDDINGS" in result["Usecase"].values

    def test_default_models_contains_correct_areas(self) -> None:
        """Should have correct area labels."""
        result = EmbeddingGenerator.list_default_models()
        areas = result["Area"].unique()
        assert "NLP" in areas
        assert "CV" in areas
        assert "STRUCTURED" in areas

    def test_default_models_dataframe_is_sorted(self) -> None:
        """Should return sorted DataFrame."""
        result = EmbeddingGenerator.list_default_models()
        # Check that the dataframe is sorted (Area, Usecase, Model Name)
        sorted_df = result.sort_values(by=list(result.columns))
        pd.testing.assert_frame_equal(result, sorted_df)

    def test_default_models_has_reset_index(self) -> None:
        """Should have reset index starting from 0."""
        result = EmbeddingGenerator.list_default_models()
        assert result.index.tolist() == list(range(len(result)))


@pytest.mark.unit
class TestListPretrainedModels:
    """Test list_pretrained_models method."""

    def test_list_pretrained_models_returns_dataframe(self) -> None:
        """Should return a pandas DataFrame."""
        result = EmbeddingGenerator.list_pretrained_models()
        assert isinstance(result, pd.DataFrame)

    def test_pretrained_models_has_required_columns(self) -> None:
        """Should have Task, Architecture, and Model Name columns."""
        result = EmbeddingGenerator.list_pretrained_models()
        assert "Task" in result.columns
        assert "Architecture" in result.columns
        assert "Model Name" in result.columns

    def test_pretrained_models_combines_nlp_and_cv(self) -> None:
        """Should include both NLP and CV models."""
        result = EmbeddingGenerator.list_pretrained_models()
        assert "NLP" in result["Task"].values
        assert "CV" in result["Task"].values

    def test_pretrained_models_has_correct_count(self) -> None:
        """Should have 16 models total (8 NLP + 8 CV)."""
        result = EmbeddingGenerator.list_pretrained_models()
        assert len(result) == 16

    def test_pretrained_models_architectures_identified(self) -> None:
        """Should identify BERT and ViT architectures."""
        result = EmbeddingGenerator.list_pretrained_models()
        architectures = result["Architecture"].unique()
        assert "BERT" in architectures
        assert "ViT" in architectures

    def test_pretrained_models_dataframe_is_sorted(self) -> None:
        """Should return sorted DataFrame."""
        result = EmbeddingGenerator.list_pretrained_models()
        sorted_df = result.sort_values(by=list(result.columns))
        pd.testing.assert_frame_equal(result, sorted_df)

    def test_pretrained_models_has_reset_index(self) -> None:
        """Should have reset index starting from 0."""
        result = EmbeddingGenerator.list_pretrained_models()
        assert result.index.tolist() == list(range(len(result)))


@pytest.mark.unit
class TestModelArchitectureParsing:
    """Test model architecture parsing."""

    def test_parse_bert_model_architecture(self) -> None:
        """Should identify BERT architecture."""
        result = EmbeddingGenerator.list_pretrained_models()
        bert_models = result[
            result["Model Name"].str.contains("bert", case=False)
        ]
        for arch in bert_models["Architecture"]:
            assert arch == "BERT"

    def test_parse_vit_model_architecture(self) -> None:
        """Should identify ViT architecture."""
        result = EmbeddingGenerator.list_pretrained_models()
        vit_models = result[
            result["Model Name"].str.contains("vit", case=False)
        ]
        for arch in vit_models["Architecture"]:
            assert arch == "ViT"

    def test_all_nlp_models_are_bert_architecture(self) -> None:
        """Should classify all NLP models as BERT architecture."""
        result = EmbeddingGenerator.list_pretrained_models()
        nlp_models = result[result["Task"] == "NLP"]
        for arch in nlp_models["Architecture"]:
            assert arch == "BERT"

    def test_all_cv_models_are_vit_architecture(self) -> None:
        """Should classify all CV models as ViT architecture."""
        result = EmbeddingGenerator.list_pretrained_models()
        cv_models = result[result["Task"] == "CV"]
        for arch in cv_models["Architecture"]:
            assert arch == "ViT"
