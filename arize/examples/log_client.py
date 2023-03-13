import concurrent.futures as cf
import os
import time
import uuid
from random import random

import numpy as np
import pandas as pd
from arize.api import Client
from arize.utils.types import Embedding, ModelTypes

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
        features["feature_" + str(i) + "_np"] = np.float_(random())
        features["feature_" + str(i) + "_np_ll"] = np.longlong(random() * 100)
    return features


def get_embedding_features():
    return {
        "image_embedding": Embedding(
            vector=np.array([1.0, 2, 3]),
            link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
        ),
        "nlp_embedding_sentence": Embedding(
            vector=pd.Series([4.0, 5.0, 6.0, 7.0]),
            data="This is a test sentence",
        ),
        "nlp_embedding_tokens": Embedding(
            vector=pd.Series([4.0, 5.0, 6.0, 7.0]),
            data=["This", "is", "a", "test", "token", "array"],
        ),
    }


def get_tags():
    return {
        "tag_str": "arize",
        "tag_double": 20.20,
        "tag_int": 0,
        "tag_bool": True,
        "tag_None": None,
    }


features = get_features(NUM_FEATURES)
embedding_features = get_embedding_features()
tags = get_tags()
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
        embedding_features=embedding_features,
        tags=tags,
    )
    resps.append(log)

end_sending = time.time_ns()
print(
    f"INFO - {ITERATIONS} requests took a total of {int(end_sending - start) / 1000000}ms to send. "
    f"Waiting for responses."
)

for future in cf.as_completed(resps):
    res = future.result()
    print(f"✅ future completed with response code {res.status_code}")
    if res.status_code != 200:
        print(f"❌ future failed with response code {res.status_code}, {res.text}")
