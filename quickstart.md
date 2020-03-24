# Quickstart

Instrument your model to log predictions and latent truth events. The logged events allow the Arize platform to generate visualizations of features, model output and prediction evaluation metrics. Additionally the platform will provide data quality monitoring and data distribution alerts, for your production models.

Start logging with the following steps.

## 1. Create your account
---
Sign up for a free account on our site [sign up page](https://app.arize.com/login?signup=true).

## 2. Get your service key
---
When you create an account, we generate a service api key. You will need this API Key and account id for logging authentication. You can view those [here](https://app.arize.com/services).

## 3. Instrument your code
---
## Python
> If you are using our python client, add a few lines to your code to log predictions and truths. Logs are sent to Arize asynchrously. 

### Install Library

Install our library in an environment using Python 3.

```bash
pip install arize
```

### Initialize Arize

Initialize `arize` at the start of your sevice using your previously created Account ID and API Key

> We suggest adding the API via as secrets or an environment variable

```python
from arize.api import AsyncAPI

arize = AsyncAPI(account_id=1234, api_key=os.environ.get('API_KEY'))
```

### Collect your model input features and labels you'd like to track

You can track all input features used to at prediction time by logging it via a string:string dictionary.

```python
labels = {
    'state': 'ca',
    'city': 'berkeley',
    'lat': '37.8717',
    'lng': '-122.2579',
    'merchant_type': 'educational',
    'charge_amount': '20.11',
    }
```

### Log Predictions
```python
arize.log(
    model_id='sample-model-v1.43.56',
    prediction_id='plED4eERDCasd9797ca34',
    prediction_value=True,
    labels=labels,
    )
```

> We automatically discover new models logged over time based on the model ID sent on each prediction.

### Log Truths
```python
arize.log(
    model_id='sample-model-v1.43.56',
    prediction_id='plED4eERDCasd9797ca34',
    truth_value=True,
    )
```
>Once a truth for a prediction is determined, you can log those to Arize and evaluate your metrics over time. What links the truch to the original prediction is the prediction_id for a model_id

That's it! Once your service is deployed and predictions are logged you'll be able to log into your Arize account and dive into your data. Slicing it by feature labels, models, time, etc.


## Other languages
---
> If you are using a different language, you'll be able to post an HTTP request to our Arize edge-servers to log your events.

### Simple post request to Arize

```bash 
curl -X POST -H "Authorization: API_KEY" "https://api.arize.com/v1/log" -d'{"prediction":{"account_id": 0, "model_id": "test_model_1", "prediction_id":"test100", "labels":{"state":"CO", "type":"restaurant"}, "prediction_value": {"binary_value": false}}}'
```