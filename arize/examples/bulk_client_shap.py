import os
import uuid
import pandas as pd
import numpy as np
import concurrent.futures as cf

from arize.api import Client
from arize.utils.types import ModelTypes

ITERATIONS = 1
NUM_RECORDS = 2

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
shap_values = pd.DataFrame(
    np.random.random(size=(NUM_RECORDS, 12)),
    columns=list("abcdefghijkl"),
)
print(shap_values)
preds = arize.bulk_log(
    model_id="example_model_id",
    model_type=ModelTypes.NUMERIC,
    model_version="v0.1",
    prediction_ids=ids,
    prediction_labels=pred_labels,
    features=features,
    feature_names_overwrite=column_overwrite,
    shap_values=shap_values,
)

for future in cf.as_completed(preds):
    res = future.result()
    print(f"future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"future failed with response code {res.status_code}, {res.text}")
