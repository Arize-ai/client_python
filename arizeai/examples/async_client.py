import random
import time
from asyncio import get_event_loop

from arizeai.api import AsyncAPI

CALLS = 100_000

arize = AsyncAPI(account_id=1234, api_key='API_KEY', uri='http://httpbin.org/delay/1')
# arize = AsyncAPI(account_id=1234, api_key='API_KEY', uri='localhost:50050')

start = time.time() * 1000
for i in range(CALLS):
    arize.log(
        model_id='sample-model-v0',
        prediction_id='abc_'+ str(int(random.random()*CALLS)),
        prediction_value=True,
        labels={'label_key':'label_value', 'label_key1': 'label_value1'}
    )

end = time.time() * 1000
print('process took {}ms'.format(end-start))