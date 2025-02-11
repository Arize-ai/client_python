import os
from datetime import datetime, timedelta

import whylogs as why

from arize.experimental.integrations.whylabs import (
    IntegrationClient,
    WhylabsProfileAdapter,
)
from arize.exporter import ArizeExportClient
from arize.pandas.logger import Client
from arize.utils.types import Environments, Metrics, ModelTypes, Schema

os.environ["ARIZE_SPACE_ID"] = "REPLACE_ME"
os.environ["ARIZE_API_KEY"] = "REPLACE_ME"
os.environ["ARIZE_DEVELOPER_KEY"] = "REPLACE_ME"
os.environ["ARIZE_MODEL_ID"] = "REPLACE_ME"
os.environ["ARIZE_DEMO_MODEL_ID"] = "REPLACE_ME"

os.environ["WHYLABS_API_KEY"] = "REPLACE_ME"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "REPLACE_ME"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "REPLACE_ME"

FEATURE_COLUMNS = [
    "annual_income",
    "delinq_2yrs",
    "dti",
    "fico_score",
    "grade",
    "home_ownership",
    "inq_last_6mths",
    "installment",
    "interest_rate",
    "loan_amount",
    "merchant_ID",
    "merchant_risk_score",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "pymnt_plan",
    "revol_bal",
    "revol_util",
    "state",
    "term",
    "time",
    "verification_status",
]


def get_time_range(days=10):
    current_date = datetime.now() + timedelta(days=1)
    start_time = (current_date - timedelta(days=days)).replace(
        hour=19, minute=0, second=0, microsecond=0
    )
    end_time = current_date.replace(
        hour=18, minute=59, second=59, microsecond=999999
    )
    return start_time, end_time


def generate_profile(start_time, end_time):
    export_client = ArizeExportClient(api_key=os.environ["ARIZE_DEVELOPER_KEY"])
    original_df = export_client.export_model_to_df(
        space_id=os.environ["ARIZE_SPACE_ID"],
        model_id=os.environ["ARIZE_DEMO_MODEL_ID"],
        environment=Environments.PRODUCTION,
        start_time=start_time,
        end_time=end_time,
    )

    results = why.log(original_df)
    profile = results.profile()
    return profile


def test_profile_adapter():
    try:
        arize_client = Client(
            space_id=os.environ["ARIZE_SPACE_ID"],
            api_key=os.environ["ARIZE_API_KEY"],
        )

        start_time, end_time = get_time_range()

        profile = generate_profile(start_time, end_time)
        profile_df = profile.view().to_pandas()

        generator = WhylabsProfileAdapter()
        synthetic_df = generator.generate(profile_df, num_rows=len(profile_df))
        print(synthetic_df)

        schema = Schema(
            feature_column_names=FEATURE_COLUMNS,
            tag_column_names=["age__tag"],
            prediction_label_column_name="categoricalPredictionLabel",
            prediction_score_column_name="scorePredictionLabel",
            actual_label_column_name="categoricalActualLabel",
            actual_score_column_name="scoreActualLabel",
        )

        response = arize_client.log(
            dataframe=synthetic_df,
            schema=schema,
            model_id=os.environ["ARIZE_MODEL_ID"],
            model_version="1.0.0",
            model_type=ModelTypes.BINARY_CLASSIFICATION,
            metrics_validation=[Metrics.CLASSIFICATION],
            validate=True,
            environment=Environments.PRODUCTION,
        )
        print("Logging Arize response:", response)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


def test_client():
    client = IntegrationClient(
        api_key=os.environ["ARIZE_API_KEY"],
        space_id=os.environ["ARIZE_SPACE_ID"],
        developer_key=os.environ["ARIZE_DEVELOPER_KEY"],
    )

    start_time, end_time = get_time_range()
    profile = generate_profile(start_time, end_time)
    schema = Schema(
        feature_column_names=FEATURE_COLUMNS,
        tag_column_names=["age__tag"],
        prediction_label_column_name="categoricalPredictionLabel",
        prediction_score_column_name="scorePredictionLabel",
        actual_label_column_name="categoricalActualLabel",
        actual_score_column_name="scoreActualLabel",
    )
    environment = Environments.PRODUCTION
    model_id = os.environ["ARIZE_MODEL_ID"]
    model_type = ModelTypes.BINARY_CLASSIFICATION

    response = client.log_profile(
        profile,
        schema,
        environment,
        model_id,
        model_type,
    )
    print("Logging Arize response:", response)

    response = client.log_dataset_profile(
        profile,
        model_id,
    )
    print("Logging Arize response:", response)


if __name__ == "__main__":
    test_profile_adapter()
    test_client()
