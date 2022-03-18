import json
import pandas as pd
import io
import boto3
from urllib.parse import urlparse
from arize.api import Client
from arize.utils.types import ModelTypes
import concurrent.futures as cf


def get_csv_output_from_s3(s3uri, file_name):
    parsed_url = urlparse(s3uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:]
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, "{}/{}".format(prefix, file_name))
    return obj.get()["Body"].read().decode("utf-8")


def lambda_handler(event, context):
    # SPACE KEY - SUPPLIED BY ARIZE
    space_key = "SPACE_KEY_HERE"
    # API KEY - GENERATED IN ARIZE ACCOUNT OR SUPPLIED
    api_key = "API_KEY_HERE"
    # Test
    print("Arize Batch Lambda v7")
    print(json.dumps(str(event)))
    print("Dumped Event")
    if event["detail"]["TransformJobStatus"] == "Completed":
        print("Found Completed")
        if (
            "ArizeMonitor" in event["detail"]["Environment"]
            and event["detail"]["Environment"]["ArizeMonitor"] == "1"
        ):
            print("Found ArizeMonitor")
            # print(json.dumps(str(event['detail'])))≈ß
            print(
                "transform name " + json.dumps(str(event["detail"]["TransformJobName"]))
            )
            print("Found transform name")
            print(
                "output path" + str(event["detail"]["TransformOutput"]["S3OutputPath"])
            )
            if "Environment" in event["detail"]:
                # Helps debugging - prints the information passed in on the event
                # print(json.dumps(str(event['detail']['Environment'])))
                # Links and Data URLS to Inputs, predictions
                output_path = event["detail"]["TransformOutput"]["S3OutputPath"]
                batch_file_noID = event["detail"]["Environment"]["batch_file"]
                features_file = event["detail"]["Environment"]["features_file"]
                model_version_id_now = event["detail"]["Environment"]["model_version"]
                batch_id = event["detail"]["Environment"]["batch_id"]
                model_name = event["detail"]["Environment"]["model_name"]
                # Process & load data for sending to Arize
                features = pd.read_csv(features_file)
                output = get_csv_output_from_s3(
                    output_path, "{}.out".format(batch_file_noID)
                )
                # Output file has both predictions and features
                column_names = list(features["features"].values) + ["predictions"]
                output_df = pd.read_csv(
                    io.StringIO(output), sep=",", names=column_names
                )
                predictions_df = output_df["predictions"]
                features_df = output_df.drop(["predictions"], axis=1)
                ## ARIZE CLIENT SETUP ##
                arize_client = Client(space_key=space_key, api_key=api_key)
                # Turn Predictions into strings - classification 1/0
                ids = pd.DataFrame([str(x) + "_" + batch_id for x in features_df.index])
                tfuture = arize_client.log_bulk_predictions(
                    model_id=model_name,
                    model_version=model_version_id_now,
                    model_type=ModelTypes.CATEGORICAL,
                    features=features_df,
                    prediction_ids=ids,
                    prediction_labels=predictions_df,
                )
                for response in cf.as_completed(tfuture):
                    res = response.result()
                    if res.status_code != 200:
                        print(
                            f"future failed with response code {res.status_code}, {res.text}"
                        )
                return {"statusCode": 200, "body": "here"}
