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
    space_key=os.environ.get("ARIZE_SPACE_KEY"),
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


def get_shap_values(features):
    shap_values = {}
    for feature in features:
        shap_values[feature] = numpy.float_(random())
    return shap_values


features = get_features(NUM_FEATURES)
shap_values = get_shap_values(features)
resps = []
start = time.time_ns()
for j in range(ITERATIONS):
    id_ = str(uuid.uuid4())
    pred = arize.log(
        model_id="example_model_id",
        model_type=ModelTypes.BINARY,
        model_version="v0.1",
        prediction_id=id_,
        prediction_label=True,
        features=features,
        shap_values=shap_values,
    )
    resps.append(pred)

end_sending = time.time_ns()
print(
    f"{ITERATIONS} requests took a total of {int(end_sending - start)/1000000}ms to send. Waiting for responses."
)

for future in cf.as_completed(resps):
    res = future.result()
    print(f"future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"future failed with response code {res.status_code}, {res.text}")
