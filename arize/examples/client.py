import os
import time
import uuid
import numpy
from random import random
import concurrent.futures as cf

from arize.api import Client
from arize.utils.types import ModelTypes

ITERATIONS = 1
NUM_FEATURES = 5

arize = Client(
    organization_key=os.environ.get("ARIZE_ORG_KEY"),
    api_key=os.environ.get("ARIZE_API_KEY"),
)


def get_features(feature_counts):
    features = {}
    for i in range(feature_counts):
        features["feature_" + str(i) + "_bool"] = True
        features["feature_" + str(i) + "_str"] = "str val"
        features["feature_" + str(i) + "_float"] = random()
        features["feature_" + str(i) + "_np"] = numpy.float_(random())
        features["feature_" + str(i) + "_np_ll"] = numpy.longlong(random() * 100)
    return features


features = get_features(NUM_FEATURES)
resps = []
start = time.time_ns()
for j in range(ITERATIONS):
    id_ = str(uuid.uuid4())
    log = arize.log(
        model_id="example_model_id",
        model_version="v0.1",
        model_type=ModelTypes.BINARY,
        prediction_id=id_,
        prediction_label=True,
        actual_label=False,
        features=features,
    )
    resps.append(log)

end_sending = time.time_ns()
print(
    f"{ITERATIONS} requests took a total of {int(end_sending - start)/1000000}ms to send. Waiting for responses."
)

for future in cf.as_completed(resps):
    res = future.result()
    print(f"future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"future failed with response code {res.status_code}, {res.text}")
