from functools import partial
from typing import Optional, cast

import pandas as pd
from arize.utils.logging import logger

from .base_generators import NLPEmbeddingGenerator
from .constants import (
    DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL,
    DEFAULT_NLP_SUMMARIZATION_MODEL,
    IMPORT_ERROR_MESSAGE,
)
from .usecases import UseCases

try:
    from datasets import Dataset
except ImportError:
    raise ImportError(IMPORT_ERROR_MESSAGE)


class EmbeddingGeneratorForNLPSequenceClassification(NLPEmbeddingGenerator):
    def __init__(self, model_name: str = DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL, **kwargs):
        super(EmbeddingGeneratorForNLPSequenceClassification, self).__init__(
            use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
            model_name=model_name,
            **kwargs,
        )

    def generate_embeddings(
        self,
        text_col: pd.Series,
        class_label_col: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Obtain embedding vectors from your text data using pre-trained large language models.

        :param text_col: a pandas Series containing the different pieces of text.
        :param class_label_col: if this column is passed, the sentence "The classification label
        is <class_label>" will be appended to the text in the `text_col`.
        :return: a pandas Series containing the embedding vectors.
        """
        if not isinstance(text_col, pd.Series):
            raise TypeError("text_col must be a pandas Series")

        self.check_invalid_index(field=text_col)

        if class_label_col is not None:
            if not isinstance(class_label_col, pd.Series):
                raise TypeError("class_label_col must be a pandas Series")
            df = pd.concat({"text": text_col, "class_label": class_label_col}, axis=1)
            prepared_text_col = df.apply(
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
        return cast(pd.DataFrame, ds.to_pandas())["embedding_vector"]


class EmbeddingGeneratorForNLPSummarization(NLPEmbeddingGenerator):
    def __init__(self, model_name: str = DEFAULT_NLP_SUMMARIZATION_MODEL, **kwargs):
        super(EmbeddingGeneratorForNLPSummarization, self).__init__(
            use_case=UseCases.NLP.SUMMARIZATION,
            model_name=model_name,
            **kwargs,
        )

    def generate_embeddings(
        self,
        text_col: pd.Series,
    ) -> pd.Series:
        """
        Obtain embedding vectors from your text data using pre-trained large language models.

        :param text_col: a pandas Series containing the different pieces of text.
        :return: a pandas Series containing the embedding vectors.
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
        return cast(pd.DataFrame, ds.to_pandas())["embedding_vector"]
