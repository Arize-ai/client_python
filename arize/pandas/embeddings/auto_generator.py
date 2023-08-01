import pandas as pd

from . import constants
from .base_generators import BaseEmbeddingGenerator
from .constants import (
    CV_PRETRAINED_MODELS,
    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
    DEFAULT_CV_OBJECT_DETECTION_MODEL,
    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
    DEFAULT_NLP_SUMMARIZATION_MODEL,
    DEFAULT_TABULAR_MODEL,
    NLP_PRETRAINED_MODELS,
)
from .cv_generators import (
    EmbeddingGeneratorForCVImageClassification,
    EmbeddingGeneratorForCVObjectDetection,
)
from .nlp_generators import (
    EmbeddingGeneratorForNLPSequenceClassification,
    EmbeddingGeneratorForNLPSummarization,
)
from .tabular_generators import EmbeddingGeneratorForTabularFeatures
from .usecases import UseCases


class EmbeddingGenerator:
    def __init__(self, **kwargs: str):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the "
            f"`{self.__class__.__name__}.from_use_case(use_case, **kwargs)` method."
        )

    @staticmethod
    def from_use_case(use_case: str, **kwargs: str) -> BaseEmbeddingGenerator:
        if use_case == UseCases.NLP.SEQUENCE_CLASSIFICATION:
            return EmbeddingGeneratorForNLPSequenceClassification(**kwargs)
        elif use_case == UseCases.NLP.SUMMARIZATION:
            return EmbeddingGeneratorForNLPSummarization(**kwargs)
        elif use_case == UseCases.CV.IMAGE_CLASSIFICATION:
            return EmbeddingGeneratorForCVImageClassification(**kwargs)
        elif use_case == UseCases.CV.OBJECT_DETECTION:
            return EmbeddingGeneratorForCVObjectDetection(**kwargs)
        elif use_case == UseCases.STRUCTURED.TABULAR_EMBEDDINGS:
            return EmbeddingGeneratorForTabularFeatures(**kwargs)
        else:
            raise ValueError(f"Invalid use case {use_case}")

    @classmethod
    def list_default_models(cls) -> pd.DataFrame:
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
        df.sort_values(by=[col for col in df.columns], ascending=True, inplace=True)
        return df.reset_index(drop=True)

    @classmethod
    def list_pretrained_models(cls) -> pd.DataFrame:
        data = {
            "Task": ["NLP" for _ in NLP_PRETRAINED_MODELS] + ["CV" for _ in CV_PRETRAINED_MODELS],
            "Architecture": [
                cls.__parse_model_arch(model)
                for model in NLP_PRETRAINED_MODELS + CV_PRETRAINED_MODELS
            ],
            "Model Name": NLP_PRETRAINED_MODELS + CV_PRETRAINED_MODELS,
        }
        df = pd.DataFrame(data)
        df.sort_values(by=[col for col in df.columns], ascending=True, inplace=True)
        return df.reset_index(drop=True)

    @staticmethod
    def __parse_model_arch(model_name: str) -> str:
        if constants.GPT.lower() in model_name.lower():
            return constants.GPT
        elif constants.BERT.lower() in model_name.lower():
            return constants.BERT
        elif constants.VIT.lower() in model_name.lower():
            return constants.VIT
        else:
            raise ValueError("Invalid model_name, unknown architecture.")
