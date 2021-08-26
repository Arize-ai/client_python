import os
import uuid
import time
import pandas as pd
import numpy as np

from arize.utils.types import ModelTypes, Environments
from arize.pandas.logger import Client, Schema

ITERATIONS = 1
NUM_RECORDS = 1000

client = Client(
    organization_key=os.environ.get("ARIZE_ORG_KEY"),
    api_key=os.environ.get("ARIZE_API_KEY"),
    uri="https://devr.arize.com/v1",
)

features = pd.DataFrame(
    np.random.randint(0, 100000000, size=(NUM_RECORDS, 12)),
    columns=list("ABCDEFGHIJKL"),
)
pred_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(NUM_RECORDS, 1)), columns=["prediction_label"])
ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(NUM_RECORDS)], columns=["prediction_id"])
inferences = pd.concat([features, pred_labels, ids], axis=1)

start = time.time_ns()
res = client.log(
    inferences,
    "/tmp/inferences.bin",
    "model_id",
    "model_version",
    None, ModelTypes.NUMERIC,
    Environments.PRODUCTION,
    Schema(prediction_id_column_name="prediction_id",
           feature_column_names=inferences.columns.drop("prediction_label", "prediction_id"),
           prediction_label_column_name="prediction_label"))

print(f"future completed with response code {res.status_code}")
if res.status_code != 200:
    print(f"future failed with response code {res.status_code}, {res.text}")

end = time.time_ns()
print(
    f"request took a total of {int(end - start)/1000000}ms to serialize and send {NUM_RECORDS} records.\n"
)
