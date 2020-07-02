import os
import time
import uuid
import pandas as pd
import numpy as np
import concurrent.futures as cf

from arize.api import Client

ITERATIONS = 1
NUM_RECORDS = 2

arize = Client(organization_key="barcelos",
               api_key=os.environ.get('ARIZE_API_KEY'),
               model_id='benchmark_bulk_client',
               model_version="v0.1",
               max_queue_bound=5000,
               max_workers=8,
               uri='https://devr.arize.com/v1',
               timeout=500)

features = pd.DataFrame(np.random.randint(0, 100000000,
                                          size=(NUM_RECORDS, 25)),
                        columns=list('ABCDEFGHIJKLMNOPQRSTUVXYZ'))
pred_labels = pd.DataFrame(
    np.random.randint(0, 100000000, size=(NUM_RECORDS, 1)))
ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(NUM_RECORDS)])

start = time.time_ns()
resps = arize.log(prediction_ids=ids,
                  prediction_labels=pred_labels,
                  features=features,
                  actual_labels=None)
end_sending = time.time_ns()
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
            f'{complete} requests completed ({failed} failed) in {int(tmp - end_sending)/1000000}ms'
        )

end = time.time_ns()
print(f'===== Benchmark Stats =====')
print(f'Total test took {round(((end - start)/1000000000),2)}s.')
print(f'Bulk API took {round((end_sending - start)/1000000000,2)}s to return.')
print(f'Futures took {round((end-end_sending)/1000000000,2)}s to resolve.')
if failed > 0:
    print(f'{complete} responses received, {failed} failed.')
print('Done.')
