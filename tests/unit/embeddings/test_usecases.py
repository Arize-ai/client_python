"""Tests for arize.embeddings.usecases module."""

import pytest

from arize.embeddings.usecases import (
    CVUseCases,
    NLPUseCases,
    TabularUseCases,
    UseCases,
)


@pytest.mark.unit
class TestNLPUseCases:
    """Test NLPUseCases enum."""

    def test_sequence_classification_exists(self) -> None:
        """Should have SEQUENCE_CLASSIFICATION member."""
        assert hasattr(NLPUseCases, "SEQUENCE_CLASSIFICATION")
        assert NLPUseCases.SEQUENCE_CLASSIFICATION is not None

    def test_summarization_exists(self) -> None:
        """Should have SUMMARIZATION member."""
        assert hasattr(NLPUseCases, "SUMMARIZATION")
        assert NLPUseCases.SUMMARIZATION is not None

    def test_enum_has_two_members(self) -> None:
        """Should have exactly 2 members."""
        assert len(list(NLPUseCases)) == 2


@pytest.mark.unit
class TestCVUseCases:
    """Test CVUseCases enum."""

    def test_image_classification_exists(self) -> None:
        """Should have IMAGE_CLASSIFICATION member."""
        assert hasattr(CVUseCases, "IMAGE_CLASSIFICATION")
        assert CVUseCases.IMAGE_CLASSIFICATION is not None

    def test_object_detection_exists(self) -> None:
        """Should have OBJECT_DETECTION member."""
        assert hasattr(CVUseCases, "OBJECT_DETECTION")
        assert CVUseCases.OBJECT_DETECTION is not None

    def test_enum_has_two_members(self) -> None:
        """Should have exactly 2 members."""
        assert len(list(CVUseCases)) == 2


@pytest.mark.unit
class TestTabularUseCases:
    """Test TabularUseCases enum."""

    def test_tabular_embeddings_exists(self) -> None:
        """Should have TABULAR_EMBEDDINGS member."""
        assert hasattr(TabularUseCases, "TABULAR_EMBEDDINGS")
        assert TabularUseCases.TABULAR_EMBEDDINGS is not None

    def test_enum_has_one_member(self) -> None:
        """Should have exactly 1 member."""
        assert len(list(TabularUseCases)) == 1


@pytest.mark.unit
class TestUseCasesContainer:
    """Test UseCases dataclass container."""

    def test_has_nlp_attribute(self) -> None:
        """Should have NLP attribute pointing to NLPUseCases."""
        assert hasattr(UseCases, "NLP")
        assert UseCases.NLP is NLPUseCases

    def test_has_cv_attribute(self) -> None:
        """Should have CV attribute pointing to CVUseCases."""
        assert hasattr(UseCases, "CV")
        assert UseCases.CV is CVUseCases

    def test_has_structured_attribute(self) -> None:
        """Should have STRUCTURED attribute pointing to TabularUseCases."""
        assert hasattr(UseCases, "STRUCTURED")
        assert UseCases.STRUCTURED is TabularUseCases

    def test_can_access_nlp_use_cases_through_container(self) -> None:
        """Should access NLP use cases through UseCases container."""
        assert (
            UseCases.NLP.SEQUENCE_CLASSIFICATION
            == NLPUseCases.SEQUENCE_CLASSIFICATION
        )
        assert UseCases.NLP.SUMMARIZATION == NLPUseCases.SUMMARIZATION

    def test_can_access_cv_use_cases_through_container(self) -> None:
        """Should access CV use cases through UseCases container."""
        assert (
            UseCases.CV.IMAGE_CLASSIFICATION == CVUseCases.IMAGE_CLASSIFICATION
        )
        assert UseCases.CV.OBJECT_DETECTION == CVUseCases.OBJECT_DETECTION

    def test_can_access_structured_use_cases_through_container(self) -> None:
        """Should access structured use cases through UseCases container."""
        assert (
            UseCases.STRUCTURED.TABULAR_EMBEDDINGS
            == TabularUseCases.TABULAR_EMBEDDINGS
        )
