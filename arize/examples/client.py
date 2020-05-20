import time
import uuid
import numpy
from random import random
import concurrent.futures as cf

from arize.api import Client

ITERATIONS = 100000
LABELS = 1

arize = Client(account_id=1,
               api_key='<API_KEY>',
               max_queue_bound=5000,
               max_workers=80,
               uri='https://dev.arize.com/v1/log')


def get_labels(label_counts):
    labels = {}
    for i in range(label_counts):
        labels['label_' + str(i) + '_bool'] = True
        labels['label_' + str(i) + '_str'] = 'str val'
        labels['label_' + str(i) + '_float'] = random()
        labels['label_' + str(i) + '_numpy'] = numpy.float_(random())
        labels['label_' + str(i) + '_numpy_longlong'] = numpy.longlong(
            random() * 100)
    return labels


labels = get_labels(LABELS)
resps = {}
start = time.time() * 1000
for j in range(ITERATIONS):
    id_ = str(uuid.uuid4())
    fut = arize.log(model_id="futures_python_benchmarks",
                    prediction_id=id_,
                    prediction_value=True,
                    labels=labels)
    resps[fut] = j
end_sending = time.time() * 1000
print(
    f'{ITERATIONS} requests took a total of {int(end_sending - start)/1000}s to send. Waiting for responses.'
)

start_wating = time.time() * 1000
complete = 0
failed = 0
for future in cf.as_completed(resps):
    complete += 1
    res = future.result()
    if res.status_code != 200:
        print(
            f'{resps[future]} future failed with response code {res.status_code}'
        )
        failed += 1
    if complete % 10000 == 0:
        tmp = time.time() * 1000
        print(
            f'{complete} requests completed ({failed} failed) in {int(tmp - start_wating)/1000}s'
        )
    del resps[future]

end = time.time() * 1000

print(f'=====Test Summary=====')
print(f'> Total test took {int((end - start)/1000)}secs.')
print(
    f'> {ITERATIONS} requests took a total of {int(end_sending - start)/1000}s to send.'
)
print(
    f'> Waiting on requests to finish took a total of {int(end - start_wating)}ms ({int(end - start_wating)/1000}s) to send.'
)
print(f'{complete} responses received, {failed} failed.')
