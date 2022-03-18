import boto3
import json
from datetime import datetime
from arize.api import Client
from arize.utils.types import ModelTypes

if __name__ != "__main__":
    runtime = boto3.client("runtime.sagemaker")


# Arize Test LambdaHandler
def lambda_handler(event, context):
    print("Lambda Handler V 1.8")
    # This shoulld match your default bucket in Jupyter Notebook
    default_bucket = "DEFAULT_BUCKET"
    # This should match key_prefix in Jupyter Notebook
    s3_schema_key_prefix = "input_schema/abalone"
    # SPACE KEY - SUPPLIED BY ARIZE
    space_key = "SPACE_KEY"
    # API KEY - GENERATED IN ARIZE ACCOUNT OR SUPPLIED
    api_key = "API_KEY"
    verbose = True
    try:
        # Get the body from the event
        data_send = event["body"]
        s3 = boto3.resource("s3")
        content_object = s3.Object(default_bucket, s3_schema_key_prefix + "/schema.txt")
        if verbose:
            print("content_object")
        if not content_object:
            print("content_object not found")
            print(f"default bucket {default_bucket}")
            print(f"{s3_schema_key_prefix}/schema.txt not found")
        if verbose:
            print("file_content")
        file_content = content_object.get()["Body"].read().decode("utf-8")
        if verbose:
            print("json_content")
        json_content = json.loads(file_content)
        query_list = []
        columns = []
        if verbose:
            print(json_content["input"])
        if verbose:
            print("event")
        if verbose:
            print(event)
        if verbose:
            print("data send")
        if verbose:
            print(data_send)
        if verbose:
            print("data_send data")
        if verbose:
            print(type(data_send))
        if verbose:
            print(json.dumps(data_send))
        data_val = json.loads(data_send)
        if verbose:
            print("data_val")
        if verbose:
            print(data_val)
        for item in json_content["input"]:
            query_list = query_list + [data_val["data"][item["name"]]]
            columns = columns + [item["name"]]
        payload = {"data": query_list}
        if verbose:
            print(payload)
        if verbose:
            print("byte array")
        data_send_byte_array = bytearray(json.dumps(payload), encoding="utf8")
        if verbose:
            print("calling runtime")
        if verbose:
            print(data_send_byte_array)
        # Call endpoint running prediction pipeline
        # EndpointName is defined at model deployment time after its trained in the Notebook
        response = runtime.invoke_endpoint(
            EndpointName="inference-pipeline-lambda-realtime-1-0",
            ContentType="application/json",
            Body=data_send_byte_array,
        )
        response_data = float(response["Body"].read())
    except:
        return "ERROR DATA - " + str(event["body"])

    # setup columns to send as labels
    # columns = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight",
    #           "shell_weight"]
    if verbose:
        print("response")
    if verbose:
        print(response_data)
    ## ARIZE CLIENT SETUP ##
    model_name = "pipeline_plus_lambda"
    model_version_id_now = "model_ver_2.0"
    # Map labels to Features
    features = {}
    for index, col in enumerate(columns):
        features[col] = str(payload["data"][index])
    # ARIZE Client
    arize_client = Client(space_key=space_key, api_key=api_key)
    # Prediction ID in this test is Random / IT SHOULD BE SOMETHING MATACHABLE FOR ACTUALS
    prediction_id = datetime.datetime.today().strftime("%m_%d_%Y__%H_%M_%S")
    tfuture = arize_client.log_prediction(
        model_id=model_name,
        model_version=model_version_id_now,
        model_type=ModelTypes.NUMERIC,
        features=features,
        prediction_id=prediction_id,
        prediction_label=response_data,
    )
    if verbose:
        print(tfuture.result())
    if verbose:
        print([features, response_data, prediction_id])
    if verbose:
        print("result is!!!!")
    if verbose:
        print(response_data)
    return json.loads('{"predict":' + str(response_data) + "}")
