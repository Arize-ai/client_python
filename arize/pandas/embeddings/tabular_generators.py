from dataclasses import dataclass
from enum import Enum, auto, unique
from functools import partial
from typing import Dict, List, Tuple, Union, cast

import pandas as pd

from arize.utils.logging import logger
from arize.utils.utils import is_list_of
from .base_generators import NLPEmbeddingGenerator
from .constants import IMPORT_ERROR_MESSAGE
from .usecases import UseCases

try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer  # type: ignore
except ImportError:
    raise ImportError(IMPORT_ERROR_MESSAGE)

TABULAR_PRETRAINED_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "xlm-roberta-base",
]


@unique
class StructuredUseCases(Enum):
    TABULAR_FEATURES = auto()


@dataclass
class UseCases:
    STRUCTURED = StructuredUseCases


class EmbeddingGeneratorForTabularFeatures(NLPEmbeddingGenerator):
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  use_case={self.use_case},\n"
            f"  model_name={self.model_name},\n"
            f"  tokenizer_max_length={self.tokenizer_max_length},\n"
            f"  tokenizer={self.tokenizer.__class__},\n"
            f"  model={self.model.__class__},\n"
            f")"
        )

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        **kwargs,
    ):
        if model_name not in TABULAR_PRETRAINED_MODELS:
            raise ValueError(
                f"model_name not supported. Check supported models with "
                f"`EmbeddingGeneratorForTabularFeatures.list_pretrained_models()`"
            )
        super(EmbeddingGeneratorForTabularFeatures, self).__init__(
            use_case=UseCases.STRUCTURED.TABULAR_FEATURES,
            model_name=model_name,
            **kwargs,
        )

    def generate_embeddings(
        self,
        df: pd.DataFrame,
        selected_columns: List[str],
        col_name_map: Dict[str, str] = {},
        return_prompt_col: bool = False,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Obtain embedding vectors from your tabular data. Prompts are generated from your
        `selected_columns` and passed to a pre-trained large language model for embedding vector
        computation.

        :param df: pandas DataFrame containing the tabular data, not all columns will be
        considered, see `selected_columns`.
        :param selected_columns: columns to be considered to construct the prompt to be passed to
        the LLM.
        :param col_name_map: mapoing between selected column names and a more verbose description of
        the name. This helps the LLM understand the features better.
        :param return_prompt_col: if set to True, an extra pandas Series will be returned
        containing the constructed prompts. Defaults to False.
        :return: a pandas Series containing the embedding vectors and, if `return_prompt_col` is
        set to True, a pandas Seres containing the prompts created from tabular features.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not is_list_of(selected_columns, str):
            raise TypeError("columns must be a list of column names (strings)")
        missing_cols = set(selected_columns).difference(df.columns)
        if missing_cols:
            raise ValueError(
                "selected_columns list must only contain columns of the dataframe. "
                f"The following columns are not found {missing_cols}"
            )

        if not isinstance(col_name_map, dict):
            raise TypeError(
                "col_name_map must be a dictionary mapping column names to new column "
                "names"
            )
        for k, v in col_name_map.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(
                    "col_name_map dictionary keys and values should be strings"
                )
        missing_cols = set(col_name_map.keys()).difference(df.columns)
        if missing_cols:
            raise ValueError(
                "col_name_map must only contain keys which are columns of the dataframe. "
                f"The following columns are not found {missing_cols}"
            )

        prompts = df.rename(columns=col_name_map).apply(
            partial(
                self.__prompt_fn,
                columns=[col_name_map.get(col, col) for col in selected_columns],
            ),
            axis=1,
        )
        ds = Dataset.from_dict({"prompt": prompts})
        ds.set_transform(partial(self.tokenize, text_feat_name="prompt"))
        logger.info("Generating embedding vectors")
        ds = ds.map(
            lambda batch: self._get_embedding_vector(
                batch, self.__get_method_for_embedding_calculation()
            ),
            batched=True,
            batch_size=self.batch_size,
        )

        if return_prompt_col:
            return cast(pd.DataFrame, ds.to_pandas())["embedding_vector"], cast(
                pd.Series, prompts
            )

        return cast(pd.DataFrame, ds.to_pandas())["embedding_vector"]

    @staticmethod
    def __prompt_fn(row: pd.DataFrame, columns: List[str]) -> str:
        return " ".join(
            f"The {col.replace('_', ' ')} is {str(row[col]).strip()}."
            for col in columns
        )

    def __get_method_for_embedding_calculation(self):
        try:
            return {
                "bert-base-uncased": "avg_token",
                "distilbert-base-uncased": "avg_token",
                "xlm-roberta-base": "cls_token",
            }[self.model_name]
        except:
            raise ValueError(f"Unsupported model_name {self.model_name}")

    @staticmethod
    def list_pretrained_models() -> pd.DataFrame:
        return pd.DataFrame({"Model Name": sorted(TABULAR_PRETRAINED_MODELS)})
