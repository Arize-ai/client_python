import os
import time
import uuid
import numpy
from random import random
import concurrent.futures as cf

from arize.api import Client

ITERATIONS = 5000
NUM_FEATURES = 5

arize = Client(organization_id=1,
               api_key=os.environ.get('ARIZE_API_KEY'),
               max_queue_bound=5000,
               max_workers=8,
               model_id="benchmark_client",
               model_version="v0.1",
               uri='https://dev.arize.com/v1')


def get_features(feature_counts):
    features = {}
    for i in range(feature_counts):
        features['feature_' + str(i) + '_bool'] = True
        features['feature_' + str(i) + '_str'] = 'str val'
        features['feature_' + str(i) + '_float'] = random()
        features['feature_' + str(i) + '_np'] = numpy.float_(random())
        features['feature_' + str(i) + '_np_ll'] = numpy.longlong(random() *
                                                                  100)
    return features


features = get_features(NUM_FEATURES)
resps = []
start = time.time_ns()
for j in range(ITERATIONS):
    id_ = str(uuid.uuid4())
    fut = arize.log(prediction_ids=id_,
                    prediction_labels=True,
                    features=features,
                    actual_labels=None)
    resps += fut
end_sending = time.time_ns()
print(
    f'{ITERATIONS} requests took a total of {int(end_sending - start)/1000000}ms to send. Waiting for responses.'
)

start_wating = time.time_ns()
complete = 0
failed = 0
for future in cf.as_completed(resps):
    complete += 1
    res = future.result()
    if res.status_code != 200:
        print(
            f'future failed with response code {res.status_code}, {res.text}')
        failed += 1
    if complete % 10000 == 0:
        tmp = time.time_ns()
        print(
            f'{complete} requests completed ({failed} failed) in {int(tmp - start_wating)/1000000}ms'
        )

end = time.time_ns()

print(f'=====Test Summary=====')
print(f'> Total test took {int((end - start)/1000000)}ms.')
print(
    f'> {ITERATIONS} requests took a total of {int(end_sending - start)/1000000}ms to send.'
)
print(
    f'> Waiting on requests to finish took a total of {int(end - start_wating)/1000000}ms to send.'
)
print(f'{complete} responses received, {failed} failed.')
