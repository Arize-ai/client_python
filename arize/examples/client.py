import random
import time
import arize
from arize.api import API

CALLS = 10

# arize = API(account_id=1234, api_key='API_KEY', uri='http://httpbin.org/delay/1')
arize = API(account_id=1234, api_key='API_KEY', uri='http://localhost:50050/v1/log')

lable = {}
for i in range(CALLS):
    lable['label_{}'.format(i)] = 'value_{}'.format(i)

start = time.time() * 1000
for i in range(CALLS):
    arize.log(
        model_id='sample-model-v0',
        prediction_id='abc_'+ str(int(random.random()*CALLS)),
        prediction_value=True,
        labels=lable
    )
end = time.time() * 1000
print('process took {}ms'.format(end-start))