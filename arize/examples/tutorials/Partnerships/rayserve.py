from ray import serve

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import uuid
import requests
import concurrent.futures as cf
from arize.api import Client
from arize.utils.types import ModelTypes

data = datasets.load_breast_cancer()
X, y = datasets.load_breast_cancer(return_X_y=True)
X, y = X.astype(np.float32), y.astype(int)
X, y = pd.DataFrame(X, columns=data["feature_names"]), pd.Series(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier().fit(X_train, y_train)

# Integration starts here
@serve.deployment(name="ArizeModel")
class ArizeModel:
    """
    Rayserve and Arize Quick-start Integration Model
    """

    def __init__(self):
        self.model = model  # change to reading a pkl file, or otherwise
        # Step 1 Save Arize client
        self.arize = Client(space_key="YOUR_SPACE_KEY", api_key="YOUR_API_KEY")
        # Step 2 Saving model metadata for passing in later
        self.model_id = "rayserve-model"
        self.model_version = "1.0"
        self.model_type = ModelTypes.BINARY

    async def __call__(self, starlette_request):
        payload = await starlette_request.json()
        # Reloading data into correct json format
        X_test = pd.read_json(payload)
        y_pred = self.model.predict(X_test)

        # Step 3 Log production to Arize
        ids_df = pd.DataFrame([str(uuid.uuid4()) for _ in range(len(X_test))])
        log_responses = self.arize.bulk_log(
            model_id=self.model_id,
            prediction_ids=ids_df,
            model_version=self.model_version,
            prediction_labels=pd.Series(y_pred),
            features=X_test,
            model_type=self.model_type,
        )

        # Record HTTP response of logging to arize
        arize_success = True
        for response in cf.as_completed(log_responses):
            status_code = response.result().status_code
            arize_success = arize_success and status_code == 200

        # Return production inferences and arize logging results
        return {"result": y_test.to_numpy(), "arize-sucessful": arize_success}


serve.start()
# Model deployment
ArizeModel.deploy()

# Simulate production setting
input = X_test.to_json()
response = requests.get("http://localhost:8000/ArizeModel", json=input)
# Display results
print(response.text)


## Visit the [Arize Blog](https://arize.com/blog) and [Resource Center](https://arize.com/resource-hub/) for more resources on ML observability and model monitoring.
## https://www.arize.com
