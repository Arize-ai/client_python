import time
import functools

from random import random

from arize.api import AsyncClient

ITERATIONS = 10
CALLS = 10
LABELS = 1

times = []

arize = AsyncClient(account_id=0,
                    api_key='<API KEY>',
                    uri='https://dev.arize.com/v1/log')


def get_labels(label_counts):
    labels = {}
    for i in range(label_counts):
        labels['label' + str(i) + 'bool'] = True
        labels['label' + str(i) + 'str'] = 'str val'
        labels['label' + str(i) + 'float'] = random()
    return labels


labels = get_labels(LABELS)

for j in range(ITERATIONS):
    start = time.time() * 1000
    for i in range(CALLS):
        arize.log(model_id=str(j),
                  prediction_id=str(i),
                  prediction_value=True,
                  labels=labels)
    end = time.time() * 1000
    times.append((end - start))
    print('{}th iteration took {}ms per request'.format(j,
                                                        (end - start) / CALLS))

print('Total itertions took on average {}ms'.format(
    functools.reduce(lambda a, b: a + b, times) / ITERATIONS))
