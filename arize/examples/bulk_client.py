import os
import uuid
import pandas as pd
import numpy as np
import concurrent.futures as cf

from arize.api import Client

ITERATIONS = 1
NUM_RECORDS = 2

arize = Client(
    organization_key=os.environ.get("ARIZE_ORG_KEY"),
    api_key=os.environ.get("ARIZE_API_KEY"),
    model_id="benchmark_bulk_client",
    model_version="v0.1",
)

features = pd.DataFrame(
    np.random.randint(0, 100000000, size=(NUM_RECORDS, 12)),
    columns=list("ABCDEFGHIJKL"),
)
pred_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(NUM_RECORDS, 1)))
ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(NUM_RECORDS)])
column_overwrite = list("abcdefghijkl")

preds = arize.log_bulk_predictions(
    prediction_ids=ids,
    prediction_labels=pred_labels,
    features=features,
    feature_names_overwrite=column_overwrite,
)
actuals = arize.log_bulk_actuals(prediction_ids=ids, actual_labels=pred_labels)
preds.extend(actuals)
for future in cf.as_completed(preds):
    res = future.result()
    print(f"future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"future failed with response code {res.status_code}, {res.text}")
