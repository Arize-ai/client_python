from .base_generators import CVEmbeddingGenerator
from .constants import (
    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
    DEFAULT_CV_OBJECT_DETECTION_MODEL,
)
from .usecases import UseCases


class EmbeddingGeneratorForCVImageClassification(CVEmbeddingGenerator):
    def __init__(
        self, model_name: str = DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL, **kwargs
    ):
        super().__init__(
            use_case=UseCases.CV.IMAGE_CLASSIFICATION,
            model_name=model_name,
            **kwargs,
        )


class EmbeddingGeneratorForCVObjectDetection(CVEmbeddingGenerator):
    def __init__(
        self, model_name: str = DEFAULT_CV_OBJECT_DETECTION_MODEL, **kwargs
    ):
        super().__init__(
            use_case=UseCases.CV.OBJECT_DETECTION,
            model_name=model_name,
            **kwargs,
        )
