"""NLP embedding generators for text classification and summarization tasks."""

import logging
from functools import partial

import pandas as pd

from arize.embeddings.base_generators import NLPEmbeddingGenerator
from arize.embeddings.constants import (
    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
    DEFAULT_NLP_SUMMARIZATION_MODEL,
    IMPORT_ERROR_MESSAGE,
)
from arize.embeddings.usecases import UseCases

try:
    from datasets import Dataset
except ImportError:
    raise ImportError(IMPORT_ERROR_MESSAGE) from None


logger = logging.getLogger(__name__)


class EmbeddingGeneratorForNLPSequenceClassification(NLPEmbeddingGenerator):
    """Embedding generator for NLP sequence classification tasks."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
        **kwargs: object,
    ) -> None:
        """Initialize the sequence classification embedding generator.

        Args:
            model_name: Name of the pre-trained NLP model.
            **kwargs: Additional arguments for model initialization.
        """
        super().__init__(
            use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
            model_name=model_name,
            **kwargs,  # type: ignore[arg-type]
        )

    def generate_embeddings(  # type: ignore[override]
        self,
        text_col: pd.Series,
        class_label_col: pd.Series | None = None,
    ) -> pd.Series:
        """Obtain embedding vectors from your text data using pre-trained large language models.

        Args:
            text_col: A pandas Series containing the different pieces of text.
            class_label_col: If this column is passed, the sentence "The classification label
                is <class_label>" will be appended to the text in the `text_col`.

        Returns:
            A pandas Series containing the embedding vectors.
        """
        if not isinstance(text_col, pd.Series):
            raise TypeError("text_col must be a pandas Series")

        self.check_invalid_index(field=text_col)

        if class_label_col is not None:
            if not isinstance(class_label_col, pd.Series):
                raise TypeError("class_label_col must be a pandas Series")
            temp_df = pd.concat(
                {"text": text_col, "class_label": class_label_col}, axis=1
            )
            prepared_text_col = temp_df.apply(
                lambda row: f" The classification label is {row['class_label']}. {row['text']}",
                axis=1,
            )
            ds = Dataset.from_dict({"text": prepared_text_col})
        else:
            ds = Dataset.from_dict({"text": text_col})

        ds.set_transform(partial(self.tokenize, text_feat_name="text"))
        logger.info("Generating embedding vectors")
        ds = ds.map(
            lambda batch: self._get_embedding_vector(batch, "cls_token"),
            batched=True,
            batch_size=self.batch_size,
        )
        result_df: pd.DataFrame = ds.to_pandas()
        return result_df["embedding_vector"]


class EmbeddingGeneratorForNLPSummarization(NLPEmbeddingGenerator):
    """Embedding generator for NLP text summarization tasks."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLP_SUMMARIZATION_MODEL,
        **kwargs: object,
    ) -> None:
        """Initialize the text summarization embedding generator.

        Args:
            model_name: Name of the pre-trained NLP model.
            **kwargs: Additional arguments for model initialization.
        """
        super().__init__(
            use_case=UseCases.NLP.SUMMARIZATION,
            model_name=model_name,
            **kwargs,  # type: ignore[arg-type]
        )

    def generate_embeddings(  # type: ignore[override]
        self,
        text_col: pd.Series,
    ) -> pd.Series:
        """Obtain embedding vectors from your text data using pre-trained large language models.

        Args:
            text_col: A pandas Series containing the different pieces of text.

        Returns:
            A pandas Series containing the embedding vectors.
        """
        if not isinstance(text_col, pd.Series):
            raise TypeError("text_col must be a pandas Series")
        self.check_invalid_index(field=text_col)

        ds = Dataset.from_dict({"text": text_col})

        ds.set_transform(partial(self.tokenize, text_feat_name="text"))
        logger.info("Generating embedding vectors")
        ds = ds.map(
            lambda batch: self._get_embedding_vector(batch, "cls_token"),
            batched=True,
            batch_size=self.batch_size,
        )
        df: pd.DataFrame = ds.to_pandas()
        return df["embedding_vector"]
