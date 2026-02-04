"""Tests for arize.embeddings.cv_generators module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from arize.embeddings.constants import (
    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
    DEFAULT_CV_OBJECT_DETECTION_MODEL,
)
from arize.embeddings.cv_generators import (
    EmbeddingGeneratorForCVImageClassification,
    EmbeddingGeneratorForCVObjectDetection,
)


@pytest.fixture
def mock_transformers() -> Generator[dict[str, Any], None, None]:
    """Mock transformers dependencies for testing."""
    with (
        patch(
            "arize.embeddings.base_generators.AutoModel.from_pretrained"
        ) as mock_model,
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

        # Setup mock processor
        mock_processor.return_value = MagicMock()

        # Setup device mocks
        mock_cuda.return_value = False
        mock_mps.return_value = False

        yield {
            "model": mock_model,
            "processor": mock_processor,
            "cuda": mock_cuda,
            "mps": mock_mps,
        }


@pytest.mark.unit
class TestEmbeddingGeneratorForCVImageClassification:
    """Test EmbeddingGeneratorForCVImageClassification class."""

    def test_initialization_with_default_model(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should initialize with default ViT model."""
        generator = EmbeddingGeneratorForCVImageClassification()
        assert generator.model_name == DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL
        assert generator.model_name == "google/vit-base-patch32-224-in21k"

    def test_initialization_with_custom_model(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should initialize with custom model name."""
        custom_model = "google/vit-large-patch16-224-in21k"
        generator = EmbeddingGeneratorForCVImageClassification(
            model_name=custom_model
        )
        assert generator.model_name == custom_model

    def test_use_case_is_image_classification(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should have IMAGE_CLASSIFICATION use case."""
        generator = EmbeddingGeneratorForCVImageClassification()
        assert generator.use_case == "CV.IMAGE_CLASSIFICATION"

    def test_inherits_from_cv_embedding_generator(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should inherit from CVEmbeddingGenerator."""
        from arize.embeddings.base_generators import CVEmbeddingGenerator

        generator = EmbeddingGeneratorForCVImageClassification()
        assert isinstance(generator, CVEmbeddingGenerator)

    def test_model_downloaded_during_initialization(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should download model during initialization."""
        _generator = EmbeddingGeneratorForCVImageClassification()
        mock_transformers["model"].assert_called_once()

    def test_image_processor_initialized(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should initialize image processor during construction."""
        generator = EmbeddingGeneratorForCVImageClassification()
        assert generator.image_processor is not None

    def test_kwargs_passed_to_parent(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should pass kwargs to parent class."""
        generator = EmbeddingGeneratorForCVImageClassification(batch_size=50)
        assert generator.batch_size == 50


@pytest.mark.unit
class TestEmbeddingGeneratorForCVObjectDetection:
    """Test EmbeddingGeneratorForCVObjectDetection class."""

    def test_initialization_with_default_model(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should initialize with default DETR model."""
        generator = EmbeddingGeneratorForCVObjectDetection()
        assert generator.model_name == DEFAULT_CV_OBJECT_DETECTION_MODEL
        assert generator.model_name == "facebook/detr-resnet-101"

    def test_initialization_with_custom_model(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should initialize with custom model name."""
        custom_model = "facebook/detr-resnet-50"
        generator = EmbeddingGeneratorForCVObjectDetection(
            model_name=custom_model
        )
        assert generator.model_name == custom_model

    def test_use_case_is_object_detection(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should have OBJECT_DETECTION use case."""
        generator = EmbeddingGeneratorForCVObjectDetection()
        assert generator.use_case == "CV.OBJECT_DETECTION"

    def test_inherits_from_cv_embedding_generator(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should inherit from CVEmbeddingGenerator."""
        from arize.embeddings.base_generators import CVEmbeddingGenerator

        generator = EmbeddingGeneratorForCVObjectDetection()
        assert isinstance(generator, CVEmbeddingGenerator)

    def test_model_downloaded_during_initialization(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should download model during initialization."""
        _generator = EmbeddingGeneratorForCVObjectDetection()
        mock_transformers["model"].assert_called_once()

    def test_image_processor_initialized(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should initialize image processor during construction."""
        generator = EmbeddingGeneratorForCVObjectDetection()
        assert generator.image_processor is not None

    def test_kwargs_passed_to_parent(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should pass kwargs to parent class."""
        generator = EmbeddingGeneratorForCVObjectDetection(batch_size=25)
        assert generator.batch_size == 25


@pytest.mark.unit
class TestCVGeneratorsComparison:
    """Test differences between CV generator classes."""

    def test_different_default_models(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should have different default models for different use cases."""
        image_class_gen = EmbeddingGeneratorForCVImageClassification()
        object_det_gen = EmbeddingGeneratorForCVObjectDetection()

        assert image_class_gen.model_name != object_det_gen.model_name

    def test_different_use_cases(
        self, mock_transformers: dict[str, Any]
    ) -> None:
        """Should have different use cases assigned."""
        image_class_gen = EmbeddingGeneratorForCVImageClassification()
        object_det_gen = EmbeddingGeneratorForCVObjectDetection()

        assert image_class_gen.use_case != object_det_gen.use_case
        assert image_class_gen.use_case == "CV.IMAGE_CLASSIFICATION"
        assert object_det_gen.use_case == "CV.OBJECT_DETECTION"
