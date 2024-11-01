import os
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Dict, List, Union, cast

import pandas as pd

import arize.pandas.embeddings.errors as err
from arize.utils.logging import logger

from .constants import IMPORT_ERROR_MESSAGE

try:
    import torch
    from datasets import Dataset
    from PIL import Image
    from transformers import (  # type: ignore
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        BatchEncoding,
    )
    from transformers.utils import logging as transformer_logging
except ImportError as e:
    raise ImportError(IMPORT_ERROR_MESSAGE) from e

transformer_logging.set_verbosity(50)
transformer_logging.enable_progress_bar()


class BaseEmbeddingGenerator(ABC):
    def __init__(
        self, use_case: Enum, model_name: str, batch_size: int = 100, **kwargs
    ):
        self.__use_case = self._parse_use_case(use_case=use_case)
        self.__model_name = model_name
        self.__device = self.select_device()
        self.__batch_size = batch_size
        logger.info(f"Downloading pre-trained model '{self.model_name}'")
        try:
            self.__model = AutoModel.from_pretrained(
                self.model_name, **kwargs
            ).to(self.device)
        except OSError as e:
            raise err.HuggingFaceRepositoryNotFound(model_name) from e
        except Exception as e:
            raise e

    def select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            logger.warning(
                "No available GPU has been detected. The use of GPU acceleration is "
                "strongly recommended. You can check for GPU availability by running "
                "`torch.cuda.is_available()` or `torch.backends.mps.is_available()`."
            )
            return torch.device("cpu")

    @property
    def use_case(self) -> str:
        return self.__use_case

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def model(self):
        return self.__model

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int) -> None:
        err_message = "New batch size should be an integer greater than 0."
        if not isinstance(new_batch_size, int):
            raise TypeError(err_message)
        elif new_batch_size <= 0:
            raise ValueError(err_message)
        else:
            self.__batch_size = new_batch_size
            logger.info(f"Batch size has been set to {new_batch_size}.")

    @staticmethod
    def _parse_use_case(use_case: Enum) -> str:
        uc_area = use_case.__class__.__name__.split("UseCases")[0]
        uc_task = use_case.name
        return f"{uc_area}.{uc_task}"

    def _get_embedding_vector(
        self, batch: Dict[str, torch.Tensor], method
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(**batch)
        # (batch_size, seq_length/or/num_tokens, hidden_size)
        if method == "cls_token":  # Select CLS token vector
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif method == "avg_token":  # Select avg token vector
            embeddings = torch.mean(outputs.last_hidden_state, 1)
        else:
            raise ValueError(f"Invalid method = {method}")
        return {"embedding_vector": embeddings.cpu().numpy().astype(float)}

    @staticmethod
    def check_invalid_index(field: Union[pd.Series, pd.DataFrame]) -> None:
        if (field.index != field.reset_index(drop=True).index).any():
            if isinstance(field, pd.DataFrame):
                raise err.InvalidIndexError("DataFrame")
            else:
                raise err.InvalidIndexError(str(field.name))

    @abstractmethod
    def __repr__(self) -> str:
        pass


class NLPEmbeddingGenerator(BaseEmbeddingGenerator):
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  use_case={self.use_case},\n"
            f"  model_name='{self.model_name}',\n"
            f"  tokenizer_max_length={self.tokenizer_max_length},\n"
            f"  tokenizer={self.tokenizer.__class__},\n"
            f"  model={self.model.__class__},\n"
            f"  batch_size={self.batch_size},\n"
            f")"
        )

    def __init__(
        self,
        use_case: Enum,
        model_name: str,
        tokenizer_max_length: int = 512,
        **kwargs,
    ):
        super().__init__(use_case=use_case, model_name=model_name, **kwargs)
        self.__tokenizer_max_length = tokenizer_max_length
        # We don't check for the tokenizer's existence since it is coupled with the corresponding model
        # We check the model's existence in `BaseEmbeddingGenerator`
        logger.info(f"Downloading tokenizer for '{self.model_name}'")
        self.__tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, model_max_length=self.tokenizer_max_length
        )

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def tokenizer_max_length(self) -> int:
        return self.__tokenizer_max_length

    def tokenize(
        self, batch: Dict[str, List[str]], text_feat_name: str
    ) -> BatchEncoding:
        return self.tokenizer(
            batch[text_feat_name],
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).to(self.device)


class CVEmbeddingGenerator(BaseEmbeddingGenerator):
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  use_case={self.use_case},\n"
            f"  model_name='{self.model_name}',\n"
            f"  image_processor={self.image_processor.__class__},\n"
            f"  model={self.model.__class__},\n"
            f"  batch_size={self.batch_size},\n"
            f")"
        )

    def __init__(self, use_case: Enum, model_name: str, **kwargs):
        super().__init__(use_case=use_case, model_name=model_name, **kwargs)
        logger.info("Downloading image processor")
        # We don't check for the image processor's existence since it is coupled with the corresponding model
        # We check the model's existence in `BaseEmbeddingGenerator`
        self.__image_processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

    @property
    def image_processor(self):
        return self.__image_processor

    @staticmethod
    def open_image(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise ValueError(f"Cannot find image {image_path}")
        return Image.open(image_path).convert("RGB")

    def preprocess_image(
        self, batch: Dict[str, List[str]], local_image_feat_name: str
    ):
        return self.image_processor(
            [
                self.open_image(image_path)
                for image_path in batch[local_image_feat_name]
            ],
            return_tensors="pt",
        ).to(self.device)

    def generate_embeddings(self, local_image_path_col: pd.Series) -> pd.Series:
        """
        Obtain embedding vectors from your image data using pre-trained image models.

        :param local_image_path_col: a pandas Series containing the local path to the images to
        be used to generate the embedding vectors.
        :return: a pandas Series containing the embedding vectors.
        """
        if not isinstance(local_image_path_col, pd.Series):
            raise TypeError(
                "local_image_path_col_name must be pandas Series object"
            )
        self.check_invalid_index(field=local_image_path_col)

        # Validate that there are no null image paths
        if local_image_path_col.isnull().any():
            raise ValueError(
                "There can't be any null values in the local_image_path_col series"
            )

        ds = Dataset.from_dict({"local_path": local_image_path_col})
        ds.set_transform(
            partial(
                self.preprocess_image,
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
