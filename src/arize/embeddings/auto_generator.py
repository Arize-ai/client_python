"""Automatic embedding generation factory for various ML use cases."""

from typing import TypeAlias

import pandas as pd

from arize.embeddings import constants
from arize.embeddings.base_generators import BaseEmbeddingGenerator
from arize.embeddings.constants import (
    CV_PRETRAINED_MODELS,
    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
    DEFAULT_CV_OBJECT_DETECTION_MODEL,
    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
    DEFAULT_NLP_SUMMARIZATION_MODEL,
    DEFAULT_TABULAR_MODEL,
    NLP_PRETRAINED_MODELS,
)
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
from arize.embeddings.usecases import (
    CVUseCases,
    NLPUseCases,
    TabularUseCases,
    UseCases,
)

UseCaseLike: TypeAlias = str | NLPUseCases | CVUseCases | TabularUseCases


class EmbeddingGenerator:
    """Factory class for creating embedding generators based on use case."""

    def __init__(self, **kwargs: str) -> None:
        """Raise error directing users to use from_use_case factory method.

        Raises:
            OSError: Always raised to prevent direct instantiation.
        """
        raise OSError(
            f"{self.__class__.__name__} is designed to be instantiated using the "
            f"`{self.__class__.__name__}.from_use_case(use_case, **kwargs)` method."
        )

    @staticmethod
    def from_use_case(
        use_case: UseCaseLike, **kwargs: object
    ) -> BaseEmbeddingGenerator:
        """Create an embedding generator for the specified use case."""
        if use_case == UseCases.NLP.SEQUENCE_CLASSIFICATION:
            return EmbeddingGeneratorForNLPSequenceClassification(**kwargs)  # type: ignore[arg-type]
        if use_case == UseCases.NLP.SUMMARIZATION:
            return EmbeddingGeneratorForNLPSummarization(**kwargs)  # type: ignore[arg-type]
        if use_case == UseCases.CV.IMAGE_CLASSIFICATION:
            return EmbeddingGeneratorForCVImageClassification(**kwargs)  # type: ignore[arg-type]
        if use_case == UseCases.CV.OBJECT_DETECTION:
            return EmbeddingGeneratorForCVObjectDetection(**kwargs)  # type: ignore[arg-type]
        if use_case == UseCases.STRUCTURED.TABULAR_EMBEDDINGS:
            return EmbeddingGeneratorForTabularFeatures(**kwargs)  # type: ignore[arg-type]
        raise ValueError(f"Invalid use case {use_case}")

    @classmethod
    def list_default_models(cls) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` of default models for each use case."""
        df = pd.DataFrame(
            {
                "Area": ["NLP", "NLP", "CV", "CV", "STRUCTURED"],
                "Usecase": [
                    UseCases.NLP.SEQUENCE_CLASSIFICATION.name,
                    UseCases.NLP.SUMMARIZATION.name,
                    UseCases.CV.IMAGE_CLASSIFICATION.name,
                    UseCases.CV.OBJECT_DETECTION.name,
                    UseCases.STRUCTURED.TABULAR_EMBEDDINGS.name,
                ],
                "Model Name": [
                    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
                    DEFAULT_NLP_SUMMARIZATION_MODEL,
                    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
                    DEFAULT_CV_OBJECT_DETECTION_MODEL,
                    DEFAULT_TABULAR_MODEL,
                ],
            }
        )
        df.sort_values(by=list(df.columns), ascending=True, inplace=True)
        return df.reset_index(drop=True)

    @classmethod
    def list_pretrained_models(cls) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` of all available pretrained models."""
        data = {
            "Task": ["NLP" for _ in NLP_PRETRAINED_MODELS]
            + ["CV" for _ in CV_PRETRAINED_MODELS],
            "Architecture": [
                cls.__parse_model_arch(model)
                for model in NLP_PRETRAINED_MODELS + CV_PRETRAINED_MODELS
            ],
            "Model Name": NLP_PRETRAINED_MODELS + CV_PRETRAINED_MODELS,
        }
        df = pd.DataFrame(data)
        df.sort_values(by=list(df.columns), ascending=True, inplace=True)
        return df.reset_index(drop=True)

    @staticmethod
    def __parse_model_arch(model_name: str) -> str:
        if constants.GPT.lower() in model_name.lower():
            return constants.GPT
        if constants.BERT.lower() in model_name.lower():
            return constants.BERT
        if constants.VIT.lower() in model_name.lower():
            return constants.VIT
        raise ValueError("Invalid model_name, unknown architecture.")
