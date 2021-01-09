<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/arize-logo-white.jpg" width="600" /><br><br>
</div>

[![Pypi](https://badge.fury.io/py/arize.svg)](https://badge.fury.io/py/arize)
![CI](https://github.com/Arize-ai/arize/workflows/CI/badge.svg)
[![Slack](https://img.shields.io/badge/slack-@arize-yellow.svg?logo=slack)](https://join.slack.com/t/arize-ai/shared_invite/zt-g9c1j1xs-aQEwOAkU4T2x5K8cqI1Xqg)

================
### Overview

A helper library to interact with Arize AI APIs

---
## Quickstart
Instrument your model to log prediction labels, human readable/debuggable features and tags, and the actual label events once the ground truth is learned. The logged events allow the Arize platform to generate visualizations of features/tags, labels and other model metadata. Additionally, the platform will provide data quality monitoring and data distribution alerts for your production models.

Start logging with the following steps.

### 1. Create your account
Sign up for a free account by reaching out to <contacts@arize.com>.

<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/Arize%20UI%20platform.jpg" /><br><br>
</div>

### 2. Get your service API key
When you create an account, we generate a service API key. You will need this API Key and your Organization ID for logging authentication.


### 3. Instrument your code
### Python Client
If you are using the Arize python client, add a few lines to your code to log predictions and actuals. Logs are sent to Arize asynchronously.

### Install Library

Install the Arize library in an environment using Python > 3.5.3.
```sh
$ pip3 install arize
```

Or clone the repo:
```sh
$ git clone https://github.com/Arize-ai/client_python.git
$ python setup.py install
```

### Initialize Python Client

Initialize `arize` at the start of your sevice using your previously created API Key and Organization ID.

> **_NOTE:_** We suggest adding the API key as a secret or an environment variable.

```python
from arize.api import Client

API_KEY = os.environ.get('ARIZE_API_KEY') #If passing api_key via env vars

arize = Client(organization_key='ARIZE_ORG_KEY', api_key=API_KEY)
```

### Collect your model input features and labels you'd like to track

#### Real-time single prediction:
For a single real-time prediction, you can track all input features used at prediction time by logging them via a key:value dictionary.

```python
features = {
    'state': 'ca',
    'city': 'berkeley',
    'merchant_name': 'Peets Coffee',
    'pos_approved': True,
    'item_count': 10,
    'merchant_type': 'coffee shop',
    'charge_amount': 20.11,
    }
```

#### Bulk predictions:
When dealing with bulk predictions, you can pass in input features, prediction/actual labels, and prediction_ids for more than one prediction via a Pandas Dataframe where df.columns contain feature names.
```python
## e.g. labels from a CSV. Labels must be 2-D data frames where df.columns correspond to the label name
features_df = pd.read_csv('path/to/file.csv')

prediction_labels_df = pd.DataFrame(np.random.randint(1, 100, size=(features.shape[0], 1)))

ids_df = pd.DataFrame([str(uuid.uuid4()) for _ in range(len(prediction_labels.index))])
```

### Log Predictions
#### Single real-time prediction:
```python
## Returns an array of concurrent.futures.Future
pred = arize.log_prediction(
    model_id='sample-model-1',
    model_version='v1.23.64', ## Optional
    prediction_id='plED4eERDCasd9797ca34',
    prediction_label=True,
    features=features,
    )

#### To confirm request future completed successfully, await for it to resolve:
## NB: This is a blocking call
response = pred.get()
res = response.result()
if res.status_code != 200:
  print(f'future failed with response code {res.status_code}, {res.text}')
```

#### Bulk upload of predictions:
```python
responses = arize.log_bulk_predictions(
    model_id='sample-model-1',
    model_version='v1.23.64', ## Optional
    prediction_ids=ids_df,
    prediction_labels=prediction_labels_df,
    features=features_df
    )
#### To confirm request futures completed successfully, await for futures to resolve:
## NB: This is a blocking call
import concurrent.futures as cf
for response in cf.as_completed(responses):
  res = response.result()
  if res.status_code != 200:
    print(f'future failed with response code {res.status_code}, {res.text}')
```

The client's log_prediction/actual function returns a single concurrent future while log_bulk_predictions/actuals returns a list of concurrent futures for asynchronous behavior. To capture the logging response, you can await the resolved futures. If you desire a fire-and-forget pattern, you can disregard the responses altogether.

We automatically discover new models logged over time based on the model ID sent on each prediction.

### Logging Actual Labels
> **_NOTE:_** Notice the prediction_id passed in matches the original prediction sent on the previous example above.
```python
response = arize.log_actual(
    model_id='sample-model-1',
    prediction_id='plED4eERDCasd9797ca34',
    actual_label=False
    )
```

#### Bulk upload of actuals:
```python
responses = arize.log_bulk_actuals(
    model_id='sample-model-1',
    prediction_ids=ids_df,
    actual_labels=actual_labels_df,
    )

#### To confirm request futures completed successfully, await for futures to resolve:
## NB: This is a blocking call
import concurrent.futures as cf
for response in cf.as_completed(responses):
  res = response.result()
  if res.status_code != 200:
    print(f'future failed with response code {res.status_code}, {res.text}')
```
Once the actual labels (ground truth) for your predictions have been determined, you can send them to Arize and evaluate your metrics over time. The prediction id for one prediction links to its corresponding actual label so it's important to note those must be the same when matching events.

### 4. Log In for Analytics
That's it! Once your service is deployed and predictions are logged you'll be able to log into your Arize account and dive into your data, slicing it by features, tags, models, time, etc.

#### Analytics Dashboard
<div align="center">
  <img src="https://storage.googleapis.com/arize-assets/Arize%20UI%20platform.jpg" /><br><br>
</div>

---
## Other languages
If you are using a different language, you'll be able to post an HTTP request to our Arize edge-servers to log your events.

### HTTP post request to Arize

```bash
curl -X POST -H "Authorization: YOU_API_KEY" "https://log.arize.com/v1/log" -d'{"organization_key": "YOUR_ORG_KEY", "model_id": "test_model_1", "prediction_id":"test100", "prediction":{"model_version": "v1.23.64", "features":{"state":{"string": "CO"}, "item_count":{"int": 10}, "charge_amt":{"float": 12.34}, "physical_card":{"string": true}}, "prediction_label": {"binary": false}}}'
```
---
