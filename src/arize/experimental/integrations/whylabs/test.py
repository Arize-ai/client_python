import os
from datetime import datetime, timedelta

import whylogs as why

from arize.experimental.integrations.whylabs import WhylabsProfileAdapter
from arize.exporter import ArizeExportClient
from arize.pandas.logger import Client
from arize.utils.types import Environments, Metrics, ModelTypes, Schema

# Configuration
CONFIG = {
    "arize": {
        "space_id": "SPACE_ID_PLACEHOLDER",
        "api_key": "API_KEY_PLACEHOLDER",
        "developer_key": "DEVELOPER_KEY_PLACEHOLDER",
        "model_id": "MODEL_ID_PLACEHOLDER",
        "synthetic_model_id": "SYNTHETIC_MODEL_ID_PLACEHOLDER",
    },
    "whylabs": {
        "api_key": "WHYLABS_API_KEY_PLACEHOLDER",
        "org_id": "ORG_ID_PLACEHOLDER",
        "dataset_id": "DATASET_ID_PLACEHOLDER",
    },
}

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


def setup_environment():
    os.environ["WHYLABS_API_KEY"] = CONFIG["whylabs"]["api_key"]
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = CONFIG["whylabs"]["org_id"]
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = CONFIG["whylabs"]["dataset_id"]
    os.environ["WHYLOGS_NO_ANALYTICS"] = "True"


def get_time_range(days=10):
    current_date = datetime.now() + timedelta(days=1)
    start_time = (current_date - timedelta(days=days)).replace(
        hour=19, minute=0, second=0, microsecond=0
    )
    end_time = current_date.replace(
        hour=18, minute=59, second=59, microsecond=999999
    )
    return start_time, end_time


def main():
    try:
        setup_environment()

        export_client = ArizeExportClient(
            api_key=CONFIG["arize"]["developer_key"]
        )
        arize_client = Client(
            space_id=CONFIG["arize"]["space_id"],
            api_key=CONFIG["arize"]["api_key"],
        )

        start_time, end_time = get_time_range()
        original_df = export_client.export_model_to_df(
            space_id=CONFIG["arize"]["space_id"],
            model_id=CONFIG["arize"]["model_id"],
            environment=Environments.PRODUCTION,
            start_time=start_time,
            end_time=end_time,
        )

        results = why.log(original_df)
        profile = results.profile()
        profile_view = profile.view()
        profile_df = profile_view.to_pandas()

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
            model_id=CONFIG["arize"]["synthetic_model_id"],
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


if __name__ == "__main__":
    main()
