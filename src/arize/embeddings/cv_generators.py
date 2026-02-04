"""Computer vision embedding generators for image classification and object detection."""

from arize.embeddings.base_generators import CVEmbeddingGenerator
from arize.embeddings.constants import (
    DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
    DEFAULT_CV_OBJECT_DETECTION_MODEL,
)
from arize.embeddings.usecases import UseCases


class EmbeddingGeneratorForCVImageClassification(CVEmbeddingGenerator):
    """Embedding generator for computer vision image classification tasks."""

    def __init__(
        self,
        model_name: str = DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL,
        **kwargs: object,
    ) -> None:
        """Initialize the image classification embedding generator.

        Args:
            model_name: Name of the pre-trained vision model.
            **kwargs: Additional arguments for model initialization.
        """
        super().__init__(
            use_case=UseCases.CV.IMAGE_CLASSIFICATION,
            model_name=model_name,
            **kwargs,  # type: ignore[arg-type]
        )


class EmbeddingGeneratorForCVObjectDetection(CVEmbeddingGenerator):
    """Embedding generator for computer vision object detection tasks."""

    def __init__(
        self,
        model_name: str = DEFAULT_CV_OBJECT_DETECTION_MODEL,
        **kwargs: object,
    ) -> None:
        """Initialize the object detection embedding generator.

        Args:
            model_name: Name of the pre-trained vision model.
            **kwargs: Additional arguments for model initialization.
        """
        super().__init__(
            use_case=UseCases.CV.OBJECT_DETECTION,
            model_name=model_name,
            **kwargs,  # type: ignore[arg-type]
        )
