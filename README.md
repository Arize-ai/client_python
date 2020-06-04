<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/arize-logo-white.jpg" width="600" /><br><br>
</div>

Arize AI [![PyPI version](https://badge.fury.io/py/arize.svg)](https://badge.fury.io/py/arize) ![CI](https://github.com/Arize-ai/arize/workflows/CI/badge.svg)
================
### Overview

A helper library to interact with Arize AI APIs

---
## Quickstart
Instrument your model to log predictions and latent truth events. The logged events allow the Arize platform to generate visualizations of features, model output and prediction evaluation metrics. Additionally the platform will provide data quality monitoring and data distribution alerts, for your production models.

Start logging with the following steps.

### 1. Create your account
Sign up for a free account by reaching out to <contacts@arize.com>.

<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/arize-home.png" /><br><br>
</div>

### 2. Get your service key
When you create an account, we generate a service api key. You will need this API Key and account id for logging authentication.


### 3. Instrument your code
### Python Client
If you are using our python client, add a few lines to your code to log predictions and truths. Logs are sent to Arize asynchrously. 

### Install Library

Install our library in an environment using Python 3.

```sh
$ pip3 install arize
```

Or clone the repo:
```sh
$ git clone https://github.com/Arize-ai/client_python.git
$ python setup.py install
```

### Initialize Python Client

Initialize `arize` at the start of your sevice using your previously created Account ID and API Key

> **_NOTE:_** We suggest adding the API KEY as secrets or an environment variable.

```python
from arize.api import Client

API_KEY = os.environ.get('ARIZE_API_KEY') #If passing api_key via env vars

arize = Client(account_id=1234, api_key=API_KEY, uri='https://dev.arize.com/v1')
```

### Collect your model input features and labels you'd like to track

#### Real-time single prediction:
For real-time single prediction process, you can track all input features used to at prediction time by logging it via a key:value dictionary.

```python
labels = {
    'state': 'ca',
    'city': 'berkeley',
    'lat': 37.8717,
    'lng': -122.2579,
    'pos_approved': True,
    'item_count': 10,
    'merchant_type': 'educational',
    'charge_amount': '20.11',
    }
```

#### Bulk predictions:
When dealing with bulk predictions, you can pass in input features, prediction/truth values, and prediction_ids via a Pandas Dataframe where df.coloumns contain label names.
```python
## e.g. labels from a CSV. Labels must be 2-D data frames where df.columns correspond to the label name
labels = pd.read_csv('path/to/file.csv')

values = pd.DataFrame(np.random.randint(1, 100, size=(labels.shape[0], 1)))

ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(len(values.index))])
```

### Log Predictions
#### Single real-time precition:
```python
## Returns an array of concurrent.futures.Future
responses = arize.log(
    model_id='sample-model-1',
    model_version='v1.23.64', ## Optional
    prediction_ids='plED4eERDCasd9797ca34',
    values=True,
    labels=labels,
    is_latent_truth=False
    )
```

#### Bulk upload:
```python
responses = arize.log(
    model_id='sample-model-1',
    model_version='v1.23.64', ## Optional
    prediction_ids=ids,
    values=values,
    labels=labels,
    is_latent_truth=False
    )
```
#### Awaiting for futures to resolve (optional):
```python
## NB: This is a blocking call
for response in responses:
  res = response.result()
```

Arize log returns a response future object for asyncronous behavior. To capture the logging response, you can await the resolved future. If you desire a fire-and-forget pattern you can disreguard the reponse altogether.

We automatically discover new models logged over time based on the model ID sent on each prediction.

### Logging Latent Truths
```python
response = arize.log(
    model_id='sample-model-1',
    prediction_ids='plED4eERDCasd9797ca34',
    values=True,
    is_latent_truth=True
    )
```
Once a truth for a prediction is determined, you can send those to Arize and evaluate your metrics over time. What links the truth to the original prediction is the prediction_id for a model_id

### 4. Log In for Analytics
That's it! Once your service is deployed and predictions are logged you'll be able to log into your Arize account and dive into your data. Slicing it by feature labels, models, time, etc.

#### Analytics Dashboard
<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/arize-home.png" /><br><br>
</div>

---
## Other languages
If you are using a different language, you'll be able to post an HTTP request to our Arize edge-servers to log your events.

### HTTP post request to Arize

```bash 
curl -X POST -H "Authorization: API_KEY" "https://dev.arize.com/v1/log" -d'{"account_id": 0, "model_id": "test_model_1", "prediction_id":"test100", "prediction":{"model_version": "v1.23.64", "labels":{"state":{"string_label": "CO"}, "item_count":{"int_label": 10}, "charge_amt":{"label_float": 12.34}, "physical_card":{"string_label": true}}, "prediction_value": {"binary_value": false}}}'
```
---
## Contributing

TBD