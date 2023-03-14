import concurrent.futures as cf
import os
import time
import uuid

import numpy as np
import pandas as pd
from arize.api import Client
from arize.utils.types import ModelTypes

ITERATIONS = 1
NUM_RECORDS = 100

arize = Client(
    space_key=os.environ.get("ARIZE_SPACE_KEY"),
    api_key=os.environ.get("ARIZE_API_KEY"),
)

features = pd.DataFrame(
    np.random.randint(0, 100000000, size=(NUM_RECORDS, 12)),
    columns=list("ABCDEFGHIJKL"),
)
pred_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(NUM_RECORDS, 1)))
ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(NUM_RECORDS)])
column_overwrite = list("abcdefghijkl")

start = time.time_ns()
preds = arize.bulk_log(
    model_id="example_model_id",
    model_version="v0.1",
    model_type=ModelTypes.NUMERIC,
    prediction_ids=ids,
    prediction_labels=pred_labels,
    features=features,
    feature_names_overwrite=column_overwrite,
    actual_labels=pred_labels,
)

end_enqueue = time.time_ns()
print(
    f"request took a total of {int(end_enqueue - start) / 1000000}ms to enqueue. "
    "Waiting for responses.\n"
)

for future in cf.as_completed(preds):
    res = future.result()
    print(f"future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"future failed with response code {res.status_code}, {res.text}")

end_sending = time.time_ns()
print(
    f"Process took a total of {int(end_sending - start) / 1000000}ms to send {NUM_RECORDS} records."
)
