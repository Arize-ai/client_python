import os
import time
import uuid

import pandas as pd
import numpy as np
import concurrent.futures as cf

from arize.api import Client
from arize.types import ModelTypes

MODELS = 1
VERSIONS = 1
BATCHES = 1
NUM_RECORDS = 1

arize = Client(
    space_key=os.environ.get("ARIZE_SPACE_KEY"),
    api_key=os.environ.get("ARIZE_API_KEY"),
)

features = pd.DataFrame(
    np.random.randint(0, 100000000, size=(NUM_RECORDS, 12)),
    columns=list("ABCDEFGHIJKL"),
)
pred_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(NUM_RECORDS, 1)))
actual_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(NUM_RECORDS, 1)))

ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(NUM_RECORDS)])
start = time.time_ns()

futures = []
for m in range(MODELS):
    model_id = f"multi_env_model_{m}"
    for v in range(VERSIONS):
        model_version = f"v0.{v}"
        preds = arize.log_bulk_predictions(
            model_id=model_id,
            model_type=ModelTypes.NUMERIC,
            model_version=model_version,
            prediction_ids=ids,
            prediction_labels=pred_labels,
            features=features,
        )
        futures.extend(preds)
        actuals = arize.log_bulk_actuals(
            model_id=model_id,
            prediction_ids=ids,
            actual_labels=actual_labels,
            model_type=ModelTypes.NUMERIC,
        )
        futures.extend(actuals)
        tFuture = arize.log_training_records(
            model_id=model_id,
            model_type=ModelTypes.NUMERIC,
            model_version=model_version,
            prediction_labels=pred_labels,
            actual_labels=actual_labels,
            features=features,
        )
        futures.extend(tFuture)
        for b in range(BATCHES):
            batch = f"batch-{b}"
            vFuture = arize.log_validation_records(
                model_id=model_id,
                model_type=ModelTypes.NUMERIC,
                model_version=model_version,
                batch_id=batch,
                prediction_labels=pred_labels,
                actual_labels=actual_labels,
                features=features,
            )
            futures.extend(vFuture)


end_enqueue = time.time_ns()
print(
    f"request took a total of {int(end_enqueue - start)/1000000}ms to queue. Waiting for response."
)

for future in cf.as_completed(futures):
    res = future.result()
    print(f"future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"future failed with response code {res.status_code}, {res.text}")

end_resp = time.time_ns()
print(f"Total time: {int(end_resp - start)/1000000}ms")
