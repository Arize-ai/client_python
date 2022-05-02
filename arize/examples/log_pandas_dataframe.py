import os
import uuid

import numpy as np
import pandas as pd
import time
from typing import List
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments, EmbeddingColumnNames

ITERATIONS = 1
NUM_RECORDS = 1000000
EMBEDDING_SIZE = 15


def main():
    client = Client(
        space_key=os.environ.get("ARIZE_SPACE_KEY"),
        api_key=os.environ.get("ARIZE_API_KEY"),
    )

    features = pd.DataFrame(
        np.random.randint(0, 100000000, size=(NUM_RECORDS, 12)),
        columns=["feature_" + x for x in list("ABCDEFGHIJKL")],
    )

    embedding_features = getEmbeddingFeaturesDataFrame(NUM_RECORDS, EMBEDDING_SIZE)

    pred_labels = pd.DataFrame(
        np.random.randint(0, 100000000, size=(NUM_RECORDS, 2)),
        columns=["prediction_label", "prediction_score"],
    )
    pred_labels["prediction_label"] = pred_labels["prediction_label"].astype(str)
    ids = pd.DataFrame(
        [str(uuid.uuid4()) for _ in range(NUM_RECORDS)], columns=["prediction_id"]
    )
    inferences = pd.concat([features, embedding_features, pred_labels, ids], axis=1)

    start = time.time_ns()
    res = client.log(
        dataframe=inferences,
        path="/tmp/arrow-inferences.bin",
        model_id="model_id",
        model_version="model_version",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=[
                col for col in inferences.columns if "feature_" in col
            ],
            embedding_feature_column_names=getEmbeddingFeaturesColumnNames(),
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="prediction_score",
        ),
    )

    print(f"✅ Future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"❌ future failed with response code {res.status_code}, {res.text}")

    end = time.time_ns()
    print(
        f" -- request took a total of {int(end - start)/1000000}ms to serialize and send {NUM_RECORDS} records.\n"
    )


# getEmbeddingFeaturesDataFrame returns a dataframe containing 3 embedding features
# + Image embeddings:
#     - vector fields are lists of floats
#     - link fields are strings
# + Sentence embeddings:
#     - vector fields are ndarrays of floats
#     - data fields are strings
# + Token array embeddings:
#     - vector fields are ndarrays of floats
#     - data fields are token arrays (lists of strings)
def getEmbeddingFeaturesDataFrame(n_records: int, emb_size: int) -> pd.DataFrame:
    return pd.DataFrame(
        dict(
            image=np.random.randn(n_records, emb_size).tolist(),
            image_link=["link_" + str(x) for x in range(n_records)],
            sentence=[np.random.randn(emb_size) for x in range(n_records)],
            sentence_data=["sentence_" + str(x) for x in range(n_records)],
            token_array=[np.random.randn(emb_size) for x in range(n_records)],
            token_array_data=[["Token", "array", str(x)] for x in range(n_records)],
        )
    )


def getEmbeddingFeaturesColumnNames() -> List[EmbeddingColumnNames]:
    return [
        EmbeddingColumnNames(
            vector_column_name="image",  # Will be name of embedding feature in the app
            link_to_data_column_name="image_link",
        ),
        EmbeddingColumnNames(
            vector_column_name="sentence",  # Will be name of embedding feature in the app
            data_column_name="sentence_data",
        ),
        EmbeddingColumnNames(
            vector_column_name="token_array",  # Will be name of embedding feature in the app
            data_column_name="token_array_data",
        ),
    ]


if __name__ == "__main__":
    main()
