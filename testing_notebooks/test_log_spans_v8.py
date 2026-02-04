import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from arize import ArizeClient

DATAFILE = "./spans_data.parquet"
SPACE_ID = "U3BhY2U6NTA3MDpsTlIr"
API_KEY = (
    "ak-ed52c99c-19ac-4f89-b43b-8c650d6771c9-KdAJlHVKJxS-ZTbc8xlAPsEo-7ZFCPIC"
)
PROJECT_NAME = "test-sdkv8-09-30-25-b"
FORCE_HTTP = False
SPAN_ID = "5b21f6d17c86fa13"  # playground
DATETIME_FMT = "%Y-%m-%d"


def main():
    os.environ["ARIZE_LOG_ENABLE"] = "true"
    os.environ["ARIZE_LOG_LEVEL"] = "debug"
    os.environ["ARIZE_LOG_STRUCTURED"] = "false"

    spans_df = get_dataframe()

    client = ArizeClient(api_key=API_KEY)
    print("arize client", client)

    # =========
    # LOG SPANS
    # =========
    response = client.spans.log(
        space_id=SPACE_ID,
        project_name=PROJECT_NAME,
        dataframe=spans_df,
    )
    if response.status_code != 200:
        print(
            f"❌ logging failed with response code {response.status_code}, {response.text}"
        )
    else:
        print("✅ You have successfully logged training set to Arize")

    # ============
    # UPDATE EVALS
    # ============

    print(f"FORCE_HTTP = {FORCE_HTTP}")
    eval_name = f"force_http_{str(FORCE_HTTP)}"

    evals_df = pd.DataFrame(
        {
            "context.span_id": [SPAN_ID],  # Use your span_id
            f"eval.{eval_name}.label": ["accuracy"],  # Example label name
            f"eval.{eval_name}.score": [0.8],  # Example label value
            f"eval.{eval_name}.explanation": ["some explanation"],
        }
    )
    response = client.spans.update_evaluations(
        space_id=SPACE_ID,
        project_name=PROJECT_NAME,
        dataframe=evals_df,
        force_http=FORCE_HTTP,
    )
    print("update evals response", response)

    # ==================
    # UPDATE ANNOTATIONS
    # ==================

    annotations_df = pd.DataFrame(
        {
            "context.span_id": [SPAN_ID],
            "annotation.quality.label": ["good"],
            "annotation.relevance.label": ["relevant"],
            "annotation.relevance.updated_by": ["human_annotator_1"],
            "annotation.sentiment_score.score": [4.5],
            "annotation.notes": ["User confirmed the summary was helpful."],
        }
    )
    response = client.spans.update_annotations(
        space_id=SPACE_ID,
        project_name=PROJECT_NAME,
        dataframe=annotations_df,
    )
    print("update annotation response", response)

    # ===============
    # UPDATE METADATA
    # ===============
    metadata_df = pd.DataFrame(
        {
            "context.span_id": [SPAN_ID],
            "attributes.metadata.status": ["reviewed"],
            "attributes.metadata.tag": ["important"],
        }
    )
    response = client.spans.update_metadata(
        space_id=SPACE_ID,
        project_name=PROJECT_NAME,
        dataframe=metadata_df,
    )
    print("update metadata response", response)

    # ============
    # EXPORT SPANS
    # ============
    start_time = datetime.strptime("2024-01-01", DATETIME_FMT)
    end_time = datetime.strptime("2026-01-01", DATETIME_FMT)
    df = client.spans.export_to_df(
        space_id=SPACE_ID,
        project_name=PROJECT_NAME,
        start_time=start_time,
        end_time=end_time,
    )
    print("export df columns", df.columns)


def get_dataframe():
    file_path = Path(DATAFILE)

    if file_path.exists() and file_path.is_file():
        import pandas as pd

        print(f"{DATAFILE} found!, loading from file")
        spans_df = pd.read_parquet(DATAFILE)
    else:
        print(
            f"{DATAFILE} now found, installing phoenix client and downloading data from phoenix cloud"
        )
        # !pip install -q arize-phoenix-client
        # Cloud Instance
        PHOENIX_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.1wP7wKDG6HGEg254UdEGPXjNy5iCODb8L1YNf60YhQk"
        PHOENIX_BASE_URL = "https://app.phoenix.arize.com/s/phoenix"
        PHOENIX_PROJECT_NAME = "playground"

        from phoenix.client import Client

        # Cloud instance with API key
        client = Client(
            base_url=PHOENIX_BASE_URL,
            api_key=PHOENIX_API_KEY,
        )

        spans_df = client.spans.get_spans_dataframe(
            project_identifier=PHOENIX_PROJECT_NAME,
            limit=1000,
        )
        spans_df.to_parquet(DATAFILE)
    return spans_df


if __name__ == "__main__":
    main()
