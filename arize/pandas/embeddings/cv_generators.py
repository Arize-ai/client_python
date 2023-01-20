from functools import partial
from typing import cast

import pandas as pd

from arize.utils.logging import logger
from .base_generators import CVEmbeddingGenerator
from .constants import IMPORT_ERROR_MESSAGE
from .usecases import UseCases

try:
    import torch
    from datasets import Dataset
except ImportError:
    raise ImportError(IMPORT_ERROR_MESSAGE)


class EmbeddingGeneratorForCVImageClassification(CVEmbeddingGenerator):
    def __init__(self, model_name: str = "google/vit-base-patch32-224-in21k", **kwargs):
        super(EmbeddingGeneratorForCVImageClassification, self).__init__(
            use_case=UseCases.CV.IMAGE_CLASSIFICATION, model_name=model_name, **kwargs
        )

    def generate_embeddings(self, local_image_path_col: pd.Series) -> pd.Series:
        """
        Obtain embedding vectors from your image data using pre-trained image models.

        :param local_image_path_col: a pandas Series containing the local path to the images to
        be used to generate the embedding vectors.
        :return: a pandas Series containing the embedding vectors.
        """
        if not isinstance(local_image_path_col, pd.Series):
            raise TypeError("local_image_path_col_name must be pandas Series object")

        # Validate that there are no null image paths
        if local_image_path_col.isnull().any():
            raise ValueError(
                f"There can't be any null values in the local_image_path_col series"
            )

        ds = Dataset.from_dict({"local_path": local_image_path_col})
        ds.set_transform(
            partial(
                self.extract_image_features,
                local_image_feat_name="local_path",
            )
        )
        logger.info("Generating embedding vectors")
        ds = ds.map(
            lambda batch: self._get_embedding_vector(batch, "avg_token"),
            batched=True,
            batch_size=self.batch_size,
        )
        return cast(pd.DataFrame, ds.to_pandas())["embedding_vector"]
