import pytest
import uuid
import pandas as pd
import numpy as np
from arize import public_pb2 as pb
from arize.utils.types import ModelTypes, Environments
from arize.pandas.logger import Client, Schema
import tempfile
from datetime import datetime


class TestClient(Client):
    def _post_file(self, path):
        pass


def test_production_raise_error():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    with pytest.raises(Exception) as execinfo:
        client.log(
            dataframe=df,
            path=temp_file.name,
            model_id="model_id",
            model_version="model_version",
            model_type=ModelTypes.NUMERIC,
            environment=Environments.PRODUCTION,
            schema=Schema(
                prediction_id_column_name="prediction_id",
                feature_column_names=list("A"),
            ),
        )
    assert execinfo.value.args[0].startswith("feature A is type ")
    assert execinfo.value.args[0].endswith(
        ", but must be one of: str, bool, float, int."
    )


def build_df(num_records: int):
    datetime_feature = pd.DataFrame([datetime.now()] * num_records, columns=["A"])
    ids = pd.DataFrame(
        [str(uuid.uuid4()) for _ in range(num_records)], columns=["prediction_id"]
    )
    return pd.concat([datetime_feature, ids], axis=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
