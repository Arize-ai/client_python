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
This guide will help you instrument your code to log model observability data. The types of data supported include prediction labels, human readable/debuggable model features and tags, actual labels (once the ground truth is learned), and other model related data. Logging model data allows you to generate powerful visualizations in the Arize platform to better understand and debug your model's behavior. Additionally, Arize can provide monitoring for the data quality, data drift, and performance of your production models.

Start logging your model data with the following steps:

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

> **_NOTE:_** We strongly suggest storing the API key as a secret or an environment variable.

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
pred = arize.log(
    model_id='sample-model-1',
    model_version='v1.23.64',
    model_type=ModelTypes.BINARY,
    prediction_id='plED4eERDCasd9797ca34',
    prediction_label=True,
    features=features,
    )

#### To confirm that the log request completed successfully, await for it to resolve:
## NB: This is a blocking call
response = pred.get()
res = response.result()
if res.status_code != 200:
  print(f'future failed with response code {res.status_code}, {res.text}')
```

#### Bulk upload of predictions:
```python
responses = arize.bulk_log(
    model_id='sample-model-1',
    model_version='v1.23.64',
    model_type=ModelTypes.BINARY,
    prediction_ids=ids_df,
    prediction_labels=prediction_labels_df,
    features=features_df
    )
#### To confirm that the log request completed successfully, await for futures to resolve:
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
response = arize.log(
    model_id='sample-model-1',
    model_type=ModelTypes.BINARY,
    prediction_id='plED4eERDCasd9797ca34',
    actual_label=False
    )
```

#### Bulk upload of actuals:
```python
responses = arize.bulk_log(
    model_id='sample-model-1',
    model_type=ModelTypes.BINARY,
    prediction_ids=ids_df,
    actual_labels=actual_labels_df,
    )

#### To confirm that the log request completed successfully, await for futures to resolve:
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
### Logging SHAP values
Log SHAP feature importances to the Arize platform to explain your model's predictions. By logging SHAP values you gain the ability to view the global feature importances of your predictions as well as the ability to perform cohort and prediction based analysis to compare feature importance values under varying conditions. For more information on SHAP and how to use SHAP with Arize, check out our [SHAP documentation](https://app.gitbook.com/@arize/s/arize-onboarding/platform-features/explainability/shap).

---
### Other languages
If you are using a different language, you'll be able to post an HTTP request to our Arize edge-servers to log your events.

### HTTP post request to Arize

```bash
curl -X POST -H "Authorization: YOU_API_KEY" "https://log.arize.com/v1/log" -d'{"organization_key": "YOUR_ORG_KEY", "model_id": "test_model_1", "prediction_id":"test100", "prediction":{"model_version": "v1.23.64", "features":{"state":{"string": "CO"}, "item_count":{"int": 10}, "charge_amt":{"float": 12.34}, "physical_card":{"string": true}}, "prediction_label": {"binary": false}}}'
```
---
