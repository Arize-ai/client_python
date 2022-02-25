import datetime
import os

import pandas as pd
import pytest
from requests import Response

import arize.pandas.validation.errors as err
from arize.pandas.logger import Client, Schema
from arize.utils.types import Environments, ModelTypes


class MockResponse(Response):
    def __init__(self, file_size, reason, status_code):
        self.file_size = file_size
        self.reason = reason
        self.status_code = status_code


class NoSendClient(Client):
    def _post_file(self, path, schema, sync, timeout):
        return MockResponse(os.path.getsize(path), "Success", 200)


def test_production_zero_errors():
    client = NoSendClient("apikey", "organizationkey")
    df = pd.DataFrame(
        {
            "prediction_id": pd.Series([0, 1, 2]),
            "prediction_timestamp": pd.Series(
                [
                    datetime.datetime.now(),
                    datetime.datetime.now() - datetime.timedelta(days=364),
                    datetime.datetime.now() + datetime.timedelta(days=364),
                ]
            ),
            "prediction_label": pd.Series(["fraud", "not fraud", "fraud"]),
            "prediction_score": pd.Series([0.2, 0.3, 0.4]),
            "actual_label": pd.Categorical(
                ["not fraud", "fraud", "not fraud"],
                ordered=True,
                categories=["fraud", "not fraud"],
            ),
            "actual_score": pd.Series([0, 1, 0]),
            "act_num_seq": pd.Series([None, None, [0, 0]]),
            #####
            "A": pd.Series([0, 1, 2]),
            "B": pd.Series([0.0, 1.0, 2.0]),
            "C": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
            "D": pd.Series([0, float("NaN"), 2]),
            "E": pd.Series([0, None, 2]),
            "F": pd.Series([None, float("NaN"), None]),
            "G": pd.Series(["foo", "bar", "baz"]),
            "H": pd.Series([True, False, True]),
            "I": pd.Categorical(
                ["a", "b", "c"], ordered=True, categories=["c", "b", "a"]
            ),
            #####
            "a": pd.Series([0, 1, 2]),
            "b": pd.Series([0.0, 1.0, 2.0]),
            "c": pd.Series([float("NaN"), float("NaN"), float("NaN")]),
            "d": pd.Series([0, float("NaN"), 2]),
            "e": pd.Series([0, None, 2]),
            "f": pd.Series([None, float("NaN"), None]),
        }
    )

    try:
        response = client.log(
            dataframe=df,
            model_id=3.14,
            model_version=1.0,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=Environments.PRODUCTION,
            schema=Schema(
                prediction_id_column_name="prediction_id",
                timestamp_column_name="prediction_timestamp",
                feature_column_names=list("ABCDEFGHI"),
                tag_column_names=list("ABCDEFGHI"),
                prediction_label_column_name="prediction_label",
                actual_label_column_name="actual_label",
                prediction_score_column_name="prediction_score",
                actual_score_column_name="actual_score",
                shap_values_column_names=dict(zip("ABCDEF", "abcdef")),
                actual_numeric_sequence_column_name="act_num_seq",
            ),
        )
    except err.ValidationFailure:
        assert False

    assert response.file_size > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
