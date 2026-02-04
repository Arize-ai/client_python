"""Tests for arize.embeddings.constants module."""

import pytest

from arize.embeddings.constants import (
    BERT,
    CV_PRETRAINED_MODELS,
    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
    DEFAULT_CV_OBJECT_DETECTION_MODEL,
    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
    DEFAULT_NLP_SUMMARIZATION_MODEL,
    DEFAULT_TABULAR_MODEL,
    GPT,
    IMPORT_ERROR_MESSAGE,
    NLP_PRETRAINED_MODELS,
    VIT,
)


@pytest.mark.unit
class TestDefaultModels:
    """Test default model name constants."""

    def test_default_nlp_sequence_classification_model(self) -> None:
        """Should have distilbert as default NLP sequence classification model."""
        assert (
            DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL
            == "distilbert-base-uncased"
        )

    def test_default_nlp_summarization_model(self) -> None:
        """Should have distilbert as default NLP summarization model."""
        assert DEFAULT_NLP_SUMMARIZATION_MODEL == "distilbert-base-uncased"

    def test_default_tabular_model(self) -> None:
        """Should have distilbert as default tabular model."""
        assert DEFAULT_TABULAR_MODEL == "distilbert-base-uncased"

    def test_default_cv_image_classification_model(self) -> None:
        """Should have vit-base-patch32 as default CV image classification model."""
        assert (
            DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL
            == "google/vit-base-patch32-224-in21k"
        )

    def test_default_cv_object_detection_model(self) -> None:
        """Should have detr-resnet-101 as default CV object detection model."""
        assert DEFAULT_CV_OBJECT_DETECTION_MODEL == "facebook/detr-resnet-101"

    def test_all_default_models_are_strings(self) -> None:
        """Should have all default models defined as strings."""
        assert isinstance(DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL, str)
        assert isinstance(DEFAULT_NLP_SUMMARIZATION_MODEL, str)
        assert isinstance(DEFAULT_TABULAR_MODEL, str)
        assert isinstance(DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL, str)
        assert isinstance(DEFAULT_CV_OBJECT_DETECTION_MODEL, str)


@pytest.mark.unit
class TestPretrainedModelLists:
    """Test pre-trained model list constants."""

    def test_nlp_pretrained_models_list_length(self) -> None:
        """Should have 8 NLP pre-trained models."""
        assert len(NLP_PRETRAINED_MODELS) == 8

    def test_nlp_pretrained_models_contains_bert_variants(self) -> None:
        """Should include BERT model variants."""
        assert "bert-base-cased" in NLP_PRETRAINED_MODELS
        assert "bert-base-uncased" in NLP_PRETRAINED_MODELS
        assert "bert-large-cased" in NLP_PRETRAINED_MODELS
        assert "bert-large-uncased" in NLP_PRETRAINED_MODELS

    def test_nlp_pretrained_models_contains_distilbert_variants(self) -> None:
        """Should include DistilBERT model variants."""
        assert "distilbert-base-cased" in NLP_PRETRAINED_MODELS
        assert "distilbert-base-uncased" in NLP_PRETRAINED_MODELS

    def test_nlp_pretrained_models_contains_xlm_roberta_variants(self) -> None:
        """Should include XLM-RoBERTa model variants."""
        assert "xlm-roberta-base" in NLP_PRETRAINED_MODELS
        assert "xlm-roberta-large" in NLP_PRETRAINED_MODELS

    def test_cv_pretrained_models_list_length(self) -> None:
        """Should have 8 CV pre-trained models."""
        assert len(CV_PRETRAINED_MODELS) == 8

    def test_cv_pretrained_models_contains_vit_variants(self) -> None:
        """Should include Vision Transformer model variants."""
        expected_models = [
            "google/vit-base-patch16-224-in21k",
            "google/vit-base-patch16-384",
            "google/vit-base-patch32-224-in21k",
            "google/vit-base-patch32-384",
            "google/vit-large-patch16-224-in21k",
            "google/vit-large-patch16-384",
            "google/vit-large-patch32-224-in21k",
            "google/vit-large-patch32-384",
        ]
        for model in expected_models:
            assert model in CV_PRETRAINED_MODELS

    def test_all_nlp_models_are_strings(self) -> None:
        """Should have all NLP model names as strings."""
        for model in NLP_PRETRAINED_MODELS:
            assert isinstance(model, str)

    def test_all_cv_models_are_strings(self) -> None:
        """Should have all CV model names as strings."""
        for model in CV_PRETRAINED_MODELS:
            assert isinstance(model, str)


@pytest.mark.unit
class TestModelArchitectures:
    """Test model architecture identifier constants."""

    def test_gpt_architecture_constant(self) -> None:
        """Should have GPT architecture constant defined."""
        assert GPT == "GPT"

    def test_bert_architecture_constant(self) -> None:
        """Should have BERT architecture constant defined."""
        assert BERT == "BERT"

    def test_vit_architecture_constant(self) -> None:
        """Should have ViT architecture constant defined."""
        assert VIT == "ViT"

    def test_architecture_constants_are_strings(self) -> None:
        """Should have all architecture constants as strings."""
        assert isinstance(GPT, str)
        assert isinstance(BERT, str)
        assert isinstance(VIT, str)


@pytest.mark.unit
class TestErrorMessages:
    """Test error message constants."""

    def test_import_error_message_exists(self) -> None:
        """Should have import error message defined."""
        assert IMPORT_ERROR_MESSAGE is not None
        assert isinstance(IMPORT_ERROR_MESSAGE, str)

    def test_import_error_message_contains_install_instructions(self) -> None:
        """Should include pip install instructions in import error message."""
        assert "pip install" in IMPORT_ERROR_MESSAGE
        assert "arize[auto-embeddings]" in IMPORT_ERROR_MESSAGE

    def test_import_error_message_mentions_extra_dependencies(self) -> None:
        """Should mention extra dependencies in import error message."""
        assert "extra dependencies" in IMPORT_ERROR_MESSAGE
