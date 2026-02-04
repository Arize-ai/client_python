"""Tabular data embedding generators for structured feature embeddings."""

import logging
from functools import partial

import pandas as pd

from arize.embeddings.base_generators import NLPEmbeddingGenerator
from arize.embeddings.constants import (
    DEFAULT_TABULAR_MODEL,
    IMPORT_ERROR_MESSAGE,
)
from arize.embeddings.usecases import UseCases
from arize.utils.types import is_list_of

try:
    from datasets import Dataset
except ImportError:
    raise ImportError(IMPORT_ERROR_MESSAGE) from None

logger = logging.getLogger(__name__)

TABULAR_PRETRAINED_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "xlm-roberta-base",
]


class EmbeddingGeneratorForTabularFeatures(NLPEmbeddingGenerator):
    """Embedding generator for tabular feature data using prompt-based LLM encoding."""

    def __repr__(self) -> str:
        """Return a string representation of the tabular embedding generator."""
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
        model_name: str = DEFAULT_TABULAR_MODEL,
        **kwargs: object,
    ) -> None:
        """Initialize the tabular features embedding generator.

        Args:
            model_name: Name of the pre-trained NLP model for tabular data.
            **kwargs: Additional arguments for model initialization.

        Raises:
            ValueError: If model_name is not in supported models list.
        """
        if model_name not in TABULAR_PRETRAINED_MODELS:
            raise ValueError(
                "model_name not supported. Check supported models with "
                "`EmbeddingGeneratorForTabularFeatures.list_pretrained_models()`"
            )
        super().__init__(
            use_case=UseCases.STRUCTURED.TABULAR_EMBEDDINGS,
            model_name=model_name,
            **kwargs,  # type: ignore[arg-type]
        )

    def generate_embeddings(  # type: ignore[override]
        self,
        df: pd.DataFrame,
        selected_columns: list[str],
        col_name_map: dict[str, str] | None = None,
        return_prompt_col: bool = False,
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Obtain embedding vectors from your tabular data.

        Prompts are generated from your `selected_columns` and passed to a pre-trained
        large language model for embedding vector computation.

        Args:
            df: Pandas DataFrame containing the tabular data. Not all columns will be
                considered, see `selected_columns`.
            selected_columns: Columns to be considered to construct the prompt to be passed to
                the LLM.
            col_name_map: Mapping between selected column names and a more verbose description of
                the name. This helps the LLM understand the features better.
            return_prompt_col: If set to True, an extra pandas Series will be returned
                containing the constructed prompts. Defaults to False.

        Returns:
            A pandas Series containing the embedding vectors and, if `return_prompt_col` is
            set to True, a pandas Series containing the prompts created from tabular features.
        """
        if col_name_map is None:
            col_name_map = {}
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.check_invalid_index(field=df)

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
                raise TypeError(
                    "col_name_map dictionary keys and values should be strings"
                )
        missing_cols = set(col_name_map.keys()).difference(df.columns)
        if missing_cols:
            raise ValueError(
                "col_name_map must only contain keys which are columns of the dataframe. "
                f"The following columns are not found {missing_cols}"
            )

        prompts: pd.Series = df.rename(columns=col_name_map).apply(
            partial(
                self.__prompt_fn,
                columns=[
                    col_name_map.get(col, col) for col in selected_columns
                ],
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

        result_df: pd.DataFrame = ds.to_pandas()
        if return_prompt_col:
            return result_df["embedding_vector"], prompts

        return result_df["embedding_vector"]

    @staticmethod
    def __prompt_fn(row: pd.DataFrame, columns: list[str]) -> str:
        return " ".join(
            f"The {col.replace('_', ' ')} is {str(row[col]).strip()}."
            for col in columns
        )

    def __get_method_for_embedding_calculation(self) -> str:
        try:
            return {
                "bert-base-uncased": "avg_token",
                "distilbert-base-uncased": "avg_token",
                "xlm-roberta-base": "cls_token",
            }[self.model_name]
        except Exception as exc:
            raise ValueError(
                f"Unsupported model_name {self.model_name}"
            ) from exc

    @staticmethod
    def list_pretrained_models() -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` of available pretrained tabular models."""
        return pd.DataFrame({"Model Name": sorted(TABULAR_PRETRAINED_MODELS)})
