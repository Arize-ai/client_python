"""Tests for arize.embeddings.base_generators module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from PIL import Image
from pytest import LogCaptureFixture

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from arize.embeddings.base_generators import (
    BaseEmbeddingGenerator,
    CVEmbeddingGenerator,
)
from arize.embeddings.errors import (
    HuggingFaceRepositoryNotFound,
    InvalidIndexError,
)
from arize.embeddings.usecases import UseCases


@pytest.fixture
def mock_torch_device() -> Generator[dict[str, Any], None, None]:
    """Mock torch device detection."""
    with (
        patch(
            "arize.embeddings.base_generators.torch.cuda.is_available"
        ) as mock_cuda,
        patch(
            "arize.embeddings.base_generators.torch.backends.mps.is_available"
        ) as mock_mps,
    ):
        mock_cuda.return_value = False
        mock_mps.return_value = False
        yield {"cuda": mock_cuda, "mps": mock_mps}


@pytest.fixture
def mock_transformers() -> Generator[dict[str, Any], None, None]:
    """Mock transformers library components."""
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
    ):
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        # Setup mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Setup mock processor
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance

        yield {
            "model": mock_model,
            "model_instance": mock_model_instance,
            "tokenizer": mock_tokenizer,
            "tokenizer_instance": mock_tokenizer_instance,
            "processor": mock_processor,
            "processor_instance": mock_processor_instance,
        }


@pytest.mark.unit
class TestBaseEmbeddingGenerator:
    """Test BaseEmbeddingGenerator abstract base class."""

    def test_cannot_instantiate_abstract_class(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should not be able to instantiate abstract base class."""
        with pytest.raises(TypeError, match="abstract"):
            BaseEmbeddingGenerator(  # type: ignore[abstract]
                use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
                model_name="test-model",
            )

    def test_select_device_cuda_when_available(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should select CUDA device when available."""
        with patch(
            "arize.embeddings.base_generators.torch.cuda.is_available"
        ) as mock_cuda:
            mock_cuda.return_value = True

            # Use concrete subclass for testing
            from arize.embeddings.nlp_generators import (
                EmbeddingGeneratorForNLPSequenceClassification,
            )

            generator = EmbeddingGeneratorForNLPSequenceClassification()
            assert str(generator.device) == "cuda"

    def test_select_device_mps_when_cuda_unavailable(
        self, mock_transformers: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should select MPS device when CUDA unavailable but MPS available."""
        with (
            patch(
                "arize.embeddings.base_generators.torch.cuda.is_available"
            ) as mock_cuda,
            patch(
                "arize.embeddings.base_generators.torch.backends.mps.is_available"
            ) as mock_mps,
        ):
            mock_cuda.return_value = False
            mock_mps.return_value = True

            from arize.embeddings.nlp_generators import (
                EmbeddingGeneratorForNLPSequenceClassification,
            )

            generator = EmbeddingGeneratorForNLPSequenceClassification()
            assert str(generator.device) == "mps"

    def test_select_device_cpu_fallback(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should select CPU device when no GPU available."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert str(generator.device) == "cpu"

    def test_device_selection_logs_warning(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
        caplog: LogCaptureFixture,
    ) -> None:
        """Should log warning when no GPU detected."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        with caplog.at_level("WARNING"):
            _generator = EmbeddingGeneratorForNLPSequenceClassification()
            assert "No available GPU" in caplog.text

    def test_device_property_readonly(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should not be able to set device property."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        with pytest.raises(AttributeError):
            generator.device = torch.device("cuda")  # type: ignore[misc]

    def test_batch_size_default_is_100(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should have default batch size of 100."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.batch_size == 100

    def test_batch_size_setter_validates_type(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should raise TypeError for non-integer batch size."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        with pytest.raises(TypeError, match="integer greater than 0"):
            generator.batch_size = "50"  # type: ignore[assignment]

    def test_batch_size_setter_validates_positive(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should raise ValueError for non-positive batch size."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        with pytest.raises(ValueError, match="integer greater than 0"):
            generator.batch_size = 0

    def test_batch_size_property_readable(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should be able to read batch size."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification(
            batch_size=50
        )
        assert generator.batch_size == 50

    def test_batch_size_setter_updates_value(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should update batch size when set."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        generator.batch_size = 200
        assert generator.batch_size == 200

    def test_parse_use_case_with_enum(self) -> None:
        """Should extract use case name from enum."""
        result = BaseEmbeddingGenerator._parse_use_case(
            UseCases.NLP.SEQUENCE_CLASSIFICATION
        )
        assert result == "NLP.SEQUENCE_CLASSIFICATION"

    def test_parse_use_case_extracts_area(self) -> None:
        """Should extract area from enum class name."""
        result = BaseEmbeddingGenerator._parse_use_case(
            UseCases.CV.IMAGE_CLASSIFICATION
        )
        assert result.startswith("CV.")

    def test_parse_use_case_returns_formatted_string(self) -> None:
        """Should return Area.TASK formatted string."""
        result = BaseEmbeddingGenerator._parse_use_case(
            UseCases.STRUCTURED.TABULAR_EMBEDDINGS
        )
        assert result == "Tabular.TABULAR_EMBEDDINGS"

    def test_check_invalid_index_raises_for_named_index(self) -> None:
        """Should raise InvalidIndexError for non-default index."""
        series = pd.Series([1, 2, 3], index=["a", "b", "c"], name="test_col")
        with pytest.raises(InvalidIndexError):
            BaseEmbeddingGenerator.check_invalid_index(series)

    def test_check_invalid_index_passes_for_range_index(self) -> None:
        """Should not raise error for default RangeIndex."""
        series = pd.Series([1, 2, 3])
        # Should not raise
        BaseEmbeddingGenerator.check_invalid_index(series)

    def test_check_invalid_index_raises_for_dataframe(self) -> None:
        """Should raise InvalidIndexError with 'DataFrame' for DataFrame input."""
        df = pd.DataFrame({"col": [1, 2, 3]}, index=["a", "b", "c"])
        with pytest.raises(InvalidIndexError) as exc_info:
            BaseEmbeddingGenerator.check_invalid_index(df)
        assert exc_info.value.field_name == "DataFrame"

    def test_model_name_property(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should return model name."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification(
            model_name="test-model"
        )
        assert generator.model_name == "test-model"

    def test_model_property_returns_model_instance(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should return the model instance."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.model is not None

    def test_use_case_property(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should return use case string."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.use_case == "NLP.SEQUENCE_CLASSIFICATION"

    def test_invalid_model_raises_huggingface_error(
        self, mock_torch_device: Generator[dict[str, Any], None, None]
    ) -> None:
        """Should raise HuggingFaceRepositoryNotFound for invalid model."""
        with patch(
            "arize.embeddings.base_generators.AutoModel.from_pretrained"
        ) as mock_model:
            mock_model.side_effect = OSError("Model not found")

            from arize.embeddings.nlp_generators import (
                EmbeddingGeneratorForNLPSequenceClassification,
            )

            with pytest.raises(HuggingFaceRepositoryNotFound):
                EmbeddingGeneratorForNLPSequenceClassification(
                    model_name="invalid-model-xyz"
                )

    def test_get_embedding_vector_cls_token_method(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should extract CLS token embeddings."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(
            2, 10, 768
        )  # batch, seq, hidden
        generator.model.return_value = mock_output

        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = generator._get_embedding_vector(batch, "cls_token")

        assert "embedding_vector" in result
        assert result["embedding_vector"].shape == (2, 768)

    def test_get_embedding_vector_avg_token_method(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should average token embeddings."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 10, 768)
        generator.model.return_value = mock_output

        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        result = generator._get_embedding_vector(batch, "avg_token")

        assert "embedding_vector" in result
        assert result["embedding_vector"].shape == (2, 768)

    def test_get_embedding_vector_invalid_method_raises(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should raise ValueError for invalid embedding method."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        batch = {"input_ids": torch.tensor([[1, 2, 3]])}

        with pytest.raises(ValueError, match="Invalid method"):
            generator._get_embedding_vector(batch, "invalid_method")


@pytest.mark.unit
class TestNLPEmbeddingGenerator:
    """Test NLPEmbeddingGenerator base class."""

    def test_tokenizer_initialized_during_construction(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should initialize tokenizer during construction."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.tokenizer is not None
        mock_transformers["tokenizer"].assert_called_once()

    def test_tokenizer_property_readonly(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should not be able to set tokenizer property."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        with pytest.raises(AttributeError):
            generator.tokenizer = MagicMock()  # type: ignore[misc]

    def test_tokenizer_max_length_default_512(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should have default max length of 512."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        assert generator.tokenizer_max_length == 512

    def test_tokenizer_max_length_custom(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should accept custom tokenizer max length."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification(
            tokenizer_max_length=256
        )
        assert generator.tokenizer_max_length == 256

    def test_tokenize_method_returns_batch_encoding(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should return BatchEncoding from tokenize."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock tokenizer return value
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        generator.tokenizer.return_value = mock_encoding

        batch = {"text": ["sample text", "another text"]}
        result = generator.tokenize(batch, "text")

        assert result is not None

    def test_tokenize_uses_padding_and_truncation(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should use padding and truncation in tokenize."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()

        # Mock tokenizer
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        generator.tokenizer.return_value = mock_encoding

        batch = {"text": ["sample text"]}
        generator.tokenize(batch, "text")

        # Check tokenizer was called with correct parameters
        generator.tokenizer.assert_called_once()
        call_kwargs = generator.tokenizer.call_args[1]
        assert call_kwargs["padding"] is True
        assert call_kwargs["truncation"] is True

    def test_repr_includes_use_case(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include use case in repr."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        repr_str = repr(generator)
        assert "use_case" in repr_str
        assert "NLP.SEQUENCE_CLASSIFICATION" in repr_str

    def test_repr_includes_model_name(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include model name in repr."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification(
            model_name="test-model"
        )
        repr_str = repr(generator)
        assert "model_name" in repr_str
        assert "test-model" in repr_str

    def test_repr_includes_tokenizer_max_length(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include tokenizer max length in repr."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification()
        repr_str = repr(generator)
        assert "tokenizer_max_length" in repr_str

    def test_repr_includes_batch_size(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include batch size in repr."""
        from arize.embeddings.nlp_generators import (
            EmbeddingGeneratorForNLPSequenceClassification,
        )

        generator = EmbeddingGeneratorForNLPSequenceClassification(
            batch_size=50
        )
        repr_str = repr(generator)
        assert "batch_size" in repr_str
        assert "50" in repr_str


@pytest.mark.unit
class TestCVEmbeddingGenerator:
    """Test CVEmbeddingGenerator base class."""

    def test_image_processor_initialized(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should initialize image processor during construction."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification()
        assert generator.image_processor is not None
        mock_transformers["processor"].assert_called_once()

    def test_image_processor_property_readonly(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should not be able to set image processor property."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification()
        with pytest.raises(AttributeError):
            generator.image_processor = MagicMock()  # type: ignore[misc]

    def test_open_image_with_valid_path(self, tmp_path: Path) -> None:
        """Should open and return PIL Image for valid path."""
        # Create a temporary image file
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)

        result = CVEmbeddingGenerator.open_image(str(img_path))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_open_image_converts_to_rgb(self, tmp_path: Path) -> None:
        """Should convert image to RGB mode."""
        # Create a grayscale image
        img_path = tmp_path / "test.png"
        img = Image.new("L", (100, 100), color=128)
        img.save(img_path)

        result = CVEmbeddingGenerator.open_image(str(img_path))
        assert result.mode == "RGB"

    def test_open_image_with_invalid_path_raises(self) -> None:
        """Should raise ValueError for invalid path."""
        with pytest.raises(ValueError, match="Cannot find image"):
            CVEmbeddingGenerator.open_image("/nonexistent/path/image.png")

    def test_open_image_is_static_method(self) -> None:
        """Should be callable as static method."""
        # Should not require instance
        assert callable(CVEmbeddingGenerator.open_image)

    def test_preprocess_image_calls_image_processor(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
        tmp_path: Path,
    ) -> None:
        """Should call image processor for preprocessing."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        # Create temporary image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100))
        img.save(img_path)

        generator = EmbeddingGeneratorForCVImageClassification()

        # Mock image processor
        mock_batch_feature = MagicMock()
        mock_batch_feature.to.return_value = mock_batch_feature
        generator.image_processor.return_value = mock_batch_feature

        batch = {"local_path": [str(img_path)]}
        _result = generator.preprocess_image(batch, "local_path")

        # Verify image processor was called
        generator.image_processor.assert_called_once()

    def test_generate_embeddings_validates_series_type(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should raise TypeError if input is not Series."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification()

        with pytest.raises(TypeError, match="pandas Series"):
            generator.generate_embeddings(["/path/to/image.png"])  # type: ignore[arg-type]

    def test_generate_embeddings_validates_no_nulls(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should raise ValueError if paths contain null values."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification()
        paths_with_null = pd.Series(["/path/image.png", None])

        with pytest.raises(ValueError, match="null values"):
            generator.generate_embeddings(paths_with_null)

    def test_repr_includes_use_case(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include use case in repr."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification()
        repr_str = repr(generator)
        assert "use_case" in repr_str
        assert "CV.IMAGE_CLASSIFICATION" in repr_str

    def test_repr_includes_model_name(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include model name in repr."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification(
            model_name="test-vit"
        )
        repr_str = repr(generator)
        assert "model_name" in repr_str
        assert "test-vit" in repr_str

    def test_repr_includes_image_processor(
        self,
        mock_torch_device: Generator[dict[str, Any], None, None],
        mock_transformers: Generator[dict[str, Any], None, None],
    ) -> None:
        """Should include image processor in repr."""
        from arize.embeddings.cv_generators import (
            EmbeddingGeneratorForCVImageClassification,
        )

        generator = EmbeddingGeneratorForCVImageClassification()
        repr_str = repr(generator)
        assert "image_processor" in repr_str
