import os
import uuid

import numpy as np
import pandas as pd
import time
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments

ITERATIONS = 1
NUM_RECORDS = 1000000

client = Client(
    organization_key=os.environ.get("ARIZE_ORG_KEY"),
    api_key=os.environ.get("ARIZE_API_KEY"),
)

features = pd.DataFrame(
    np.random.randint(0, 100000000, size=(NUM_RECORDS, 12)),
    columns=list("ABCDEFGHIJKL"),
)
pred_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(NUM_RECORDS, 2)), columns=["prediction_label", "prediction_score"])
pred_labels['prediction_label'] = pred_labels['prediction_label'].astype(str)
ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(NUM_RECORDS)], columns=["prediction_id"])
inferences = pd.concat([features, pred_labels, ids], axis=1)

start = time.time_ns()
res = client.log(
    dataframe=inferences,
    path="/tmp/arrow-inferences.bin",
    model_id="model_id",
    model_version="model_version",
    model_type=ModelTypes.SCORE_CATEGORICAL,
    environment=Environments.PRODUCTION,
    schema=Schema(prediction_id_column_name="prediction_id",
           feature_column_names=inferences.columns.drop("prediction_label", "prediction_id"),
           prediction_label_column_name="prediction_label",
           prediction_score_column_name="prediction_score"))

print(f"future completed with response code {res.status_code}")
if res.status_code != 200:
    print(f"future failed with response code {res.status_code}, {res.text}")

end = time.time_ns()
print(
    f"request took a total of {int(end - start)/1000000}ms to serialize and send {NUM_RECORDS} records.\n"
)
