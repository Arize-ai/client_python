import os
import uuid
from datetime import datetime

import pandas as pd

from arize import ArizeClient
from arize.types import (
    EmbeddingColumnNames,
    Environments,
    ModelTypes,
    ObjectDetectionColumnNames,
    Schema,
)

SPACE_ID = "U3BhY2U6NTA3MDpsTlIr"
API_KEY = (
    "ak-ed52c99c-19ac-4f89-b43b-8c650d6771c9-KdAJlHVKJxS-ZTbc8xlAPsEo-7ZFCPIC"
)

MODEL_NAME = "test-sdkv8-batch-09-30-25-d"
MODEL_VERSION = "1.0"
DATETIME_FMT = "%Y-%m-%d"


def main():
    os.environ["ARIZE_LOG_ENABLE"] = "true"
    os.environ["ARIZE_LOG_LEVEL"] = "debug"
    os.environ["ARIZE_LOG_STRUCTURED"] = "false"

    # =============
    # DOWNLOAD DATA
    # =============
    url = "https://storage.googleapis.com/arize-assets/fixtures/Embeddings/arize-demo-models-data/CV/Object-Detection/coco_detection_quality_drift"
    train_df = pd.read_parquet(f"{url}_training.parquet")
    prod_df = pd.read_parquet(f"{url}_production.parquet")

    last_ts = max(prod_df["prediction_ts"])
    now_ts = datetime.timestamp(datetime.now())
    delta_ts = now_ts - last_ts

    train_df["prediction_ts"] = (train_df["prediction_ts"] + delta_ts).astype(
        float
    )
    prod_df["prediction_ts"] = (prod_df["prediction_ts"] + delta_ts).astype(
        float
    )
    train_df["prediction_id"] = add_prediction_id(train_df)
    prod_df["prediction_id"] = add_prediction_id(prod_df)

    client = ArizeClient(api_key=API_KEY)
    print("arize client", client)

    schema = get_schema()

    # =================
    # LOG TRAINING DATA
    # =================
    response = client.models.log_batch(
        space_id=SPACE_ID,
        model_name=MODEL_NAME,
        model_type=ModelTypes.OBJECT_DETECTION,
        dataframe=train_df,
        schema=schema,
        environment=Environments.TRAINING,
        model_version=MODEL_VERSION,
        surrogate_explainability=True,
    )
    # If successful, the server will return a status_code of 200
    if response.status_code != 200:
        print(
            f"❌ logging failed with response code {response.status_code}, {response.text}"
        )
    else:
        print("✅ You have successfully logged training set to Arize")

    # ===================
    # LOG PRODUCTION DATA
    # ===================
    response = client.models.log_batch(
        space_id=SPACE_ID,
        model_name=MODEL_NAME,
        model_type=ModelTypes.OBJECT_DETECTION,
        dataframe=prod_df,
        schema=schema,
        environment=Environments.PRODUCTION,
        model_version=MODEL_VERSION,
        surrogate_explainability=True,
    )
    # If successful, the server will return a status_code of 200
    if response.status_code != 200:
        print(
            f"❌ logging failed with response code {response.status_code}, {response.text}"
        )
    else:
        print("✅ You have successfully logged training set to Arize")

    # ===========
    # EXPORT DATA
    # ===========
    start_time = datetime.strptime("2024-01-01", DATETIME_FMT)
    end_time = datetime.strptime("2026-01-01", DATETIME_FMT)

    df = client.models.export_to_df(
        space_id=SPACE_ID,
        model_name=MODEL_NAME,
        environment=Environments.TRAINING,
        model_version=MODEL_VERSION,
        start_time=start_time,
        end_time=end_time,
    )
    print("export df columns", df.columns)


def get_schema():
    tags = ["drift_type"]
    embedding_feature_column_names = {
        "image_embedding": EmbeddingColumnNames(
            vector_column_name="image_vector", link_to_data_column_name="url"
        )
    }
    object_detection_prediction_column_names = ObjectDetectionColumnNames(
        bounding_boxes_coordinates_column_name="prediction_bboxes",
        categories_column_name="prediction_categories",
        scores_column_name="prediction_scores",
    )
    object_detection_actual_column_names = ObjectDetectionColumnNames(
        bounding_boxes_coordinates_column_name="actual_bboxes",
        categories_column_name="actual_categories",
    )

    # Define a Schema() object for Arize to pick up data from the correct columns for logging
    return Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_ts",
        tag_column_names=tags,
        embedding_feature_column_names=embedding_feature_column_names,
        object_detection_prediction_column_names=object_detection_prediction_column_names,
        object_detection_actual_column_names=object_detection_actual_column_names,
    )


def add_prediction_id(df):
    return [str(uuid.uuid4()) for _ in range(df.shape[0])]


if __name__ == "__main__":
    main()
