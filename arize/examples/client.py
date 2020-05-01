import time
import uuid
from random import random

from arize.api import Client

ITERATIONS = 100
LABELS = 100

arize = Client(account_id=1,
               api_key='<API_KEY>',
               uri='https://dev.arize.com/v1/log')


def get_labels(label_counts):
    labels = {}
    for i in range(label_counts):
        labels['label' + str(i) + 'bool'] = True
        labels['label' + str(i) + 'str'] = 'str val'
        labels['label' + str(i) + 'float'] = random()
    return labels


labels = get_labels(LABELS)
resps = {}
start = time.time() * 1000
for j in range(ITERATIONS):
    fut = arize.log(model_id="futures_python",
                    prediction_id=str(uuid.uuid4()),
                    prediction_value=True,
                    labels=labels)
    resps[j] = fut
end = time.time() * 1000
print('{}th iteration took {}ms per request'.format(j,
                                                    (end - start) / ITERATIONS))

for idx, resp in resps.items():
    print(f'request #{idx} response: {resp.result()}')
