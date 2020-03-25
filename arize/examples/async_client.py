import random
import time
import functools 

from asyncio import get_event_loop

from arize.api import AsyncClient

ITERATIONS = 10
CALLS = 10
times = []

arize = AsyncClient(account_id=0, api_key='0000', uri='https://dev.arize.com/v1/log')

for j in range(ITERATIONS) :
    start = time.time() * 1000
    for i in range(CALLS):
        arize.log(
            model_id=str(j),
            prediction_id=str(i),
            prediction_value=True,
            labels={'label_key':'label_value', 'label_key1': 'label_value1'}
        )
    end = time.time() * 1000
    times.append((end-start))
    print('{}th iteration took {}ms per request'.format(j, (end-start)/CALLS))

print('Total itertions took on average {}ms'.format(functools.reduce(lambda a,b: a+b, times)/ITERATIONS))
