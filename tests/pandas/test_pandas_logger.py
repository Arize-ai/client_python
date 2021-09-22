import pytest
import tempfile
import uuid

import numpy as np
import pandas as pd
from arize.pandas.logger import Client, Schema
from arize.utils.types import Environments, ModelTypes

from arize import model
from arize import public_pb2 as pb


class TestClient(Client):
    def _post_file(self, path):
        pass


def test_production_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    client.log(
        dataframe=df,
        path=temp_file.name,
        model_id="model_id",
        model_version="model_version",
        model_type=ModelTypes.NUMERIC,
        environment=Environments.PRODUCTION,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKLM"),
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            shap_values_column_names={"A": "a", "B": "b", "C": "c", "D": "d"},
        ),
    )

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.PRODUCTION == header.environment

    def assert_func(rec):
        assert rec.prediction_id is not None
        assert rec.prediction.label is not None
        assert rec.actual.label is not None
        assert rec.prediction.features is not None
        assert len(rec.prediction.features) == 13
        assert len(rec.feature_importances.feature_importances) == 4

    assert_records(temp_file, assert_func, rec_func, num_records)


def test_production_score_categorical_scorecatgory_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    df['prediction_score'] = np.random.randint(0, 100000000, size=(num_records, 1))
    df['prediction_label'] = df['prediction_label'].astype(str)
    df['actual_score'] = np.random.randint(0, 100000000, size=(num_records, 1))
    df['actual_label'] = df['actual_label'].astype(str)

    client.log(
        dataframe=df,
        path=temp_file.name,
        model_id="model_id",
        model_version="model_version",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKLM"),
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="prediction_score",
            actual_label_column_name="actual_label",
            actual_score_column_name="actual_score",
            shap_values_column_names={'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}))

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.PRODUCTION == header.environment

    def assert_func(rec):
        assert rec.prediction_id is not None
        assert rec.prediction.label.score_categorical.HasField("score_category")
        assert rec.actual.label.score_categorical.HasField("score_category")
        assert rec.prediction.features is not None
        assert len(rec.prediction.features) == 13
        assert len(rec.feature_importances.feature_importances) == 4
        
    assert_records(temp_file, assert_func, rec_func, num_records)

def test_production_score_categorical_category_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    df['prediction_label'] = df['prediction_label'].astype(str)
    df['actual_label'] = df['actual_label'].astype(str)

    client.log(
        dataframe=df,
        path=temp_file.name,
        model_id="model_id",
        model_version="model_version",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKLM"),
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            shap_values_column_names={'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}))

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.PRODUCTION == header.environment

    def assert_func(rec):
        assert rec.prediction_id is not None
        print(rec.prediction.label.score_categorical.score_category.category)
        assert rec.actual.label.score_categorical.HasField("category")
        assert rec.prediction.label.score_categorical.HasField("category")
        assert rec.prediction.features is not None
        assert len(rec.prediction.features) == 13
        assert len(rec.feature_importances.feature_importances) == 4

    assert_records(temp_file, assert_func, rec_func, num_records)

def test_production_score_categorical_mixed_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    df['prediction_score'] = np.random.randint(0, 100000000, size=(num_records, 1))
    df['prediction_label'] = df['prediction_label'].astype(str)
    df['actual_label'] = df['actual_label'].astype(str)

    client.log(
        dataframe=df,
        path=temp_file.name,
        model_id="model_id",
        model_version="model_version",
        model_type=ModelTypes.SCORE_CATEGORICAL,
        environment=Environments.PRODUCTION,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKLM"),
            prediction_label_column_name="prediction_label",
            prediction_score_column_name="prediction_score",
            actual_label_column_name="actual_label",
            shap_values_column_names={'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}))

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.PRODUCTION == header.environment

    def assert_func(rec):
        assert rec.prediction_id is not None
        print(rec.prediction.label.score_categorical.score_category.category)
        assert rec.actual.label.score_categorical.HasField("category")
        assert rec.prediction.label.score_categorical.HasField("score_category")
        assert rec.prediction.features is not None
        assert len(rec.prediction.features) == 13
        assert len(rec.feature_importances.feature_importances) == 4
    
    assert_records(temp_file, assert_func, rec_func, num_records)

def test_training_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    client.log(
        dataframe=df,
        path=temp_file.name,
        model_id="model_id",
        model_version="model_version",
        model_type=ModelTypes.NUMERIC,
        environment=Environments.TRAINING,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKLM"),
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
        ),
    )

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.TRAINING == header.environment

    def assert_func(rec):
        assert rec.training_record.record.prediction_id is not None
        assert rec.training_record.record.prediction.label is not None
        assert rec.training_record.record.actual.label is not None
        assert rec.training_record.record.prediction.features is not None
        assert len(rec.training_record.record.prediction.features) == 13

    assert_records(temp_file, assert_func, preprod_rec_func, num_records)


def test_validation_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    client.log(
        dataframe=df,
        path=temp_file.name,
        model_id="model_id",
        model_version="model_version",
        batch_id="batch_id",
        model_type=ModelTypes.NUMERIC,
        environment=Environments.VALIDATION,
        schema=Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKLM"),
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
        ),
    )

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.VALIDATION == header.environment

    def assert_func(rec):
        assert rec.validation_record.batch_id == "batch_id"
        assert rec.validation_record.record.prediction_id is not None
        assert rec.validation_record.record.prediction.label is not None
        assert rec.validation_record.record.actual.label is not None
        assert rec.validation_record.record.prediction.features is not None
        assert len(rec.validation_record.record.prediction.features) == 13

    assert_records(temp_file, assert_func, preprod_rec_func, num_records)


def get_header(f):
    sz = int.from_bytes(f.read(8), "big", signed=False)
    header = pb.FileHeader()
    header.ParseFromString(f.read(sz))
    return header


def build_df(num_records: int):
    features = pd.DataFrame(
        np.random.randint(0, 100000000, size=(num_records, 12)),
        columns=list("ABCDEFGHIJKL"),
    )
    bool_feature = pd.DataFrame(
        np.random.choice(a=[False, True], size=(num_records, 1)), columns=["M"]
    )
    pred_labels = pd.DataFrame(
        np.random.randint(0, 100000000, size=(num_records, 1)),
        columns=["prediction_label"],
    )
    actuals_labels = pd.DataFrame(
        np.random.randint(0, 100000000, size=(num_records, 1)), columns=["actual_label"]
    )
    fi = pd.DataFrame(
        np.random.randint(0, 100000000, size=(num_records, 12)),
        columns=list("abcdefghijkl"),
    )
    ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(num_records)], columns=["prediction_id"])
    return pd.concat([features, bool_feature, pred_labels, actuals_labels, ids, fi], axis=1)

  
def preprod_rec_func():
    return pb.PreProductionRecord()

  
def rec_func():
    return pb.Record()

  
def assert_records(temp_file, assert_func, record_func, num_records):
    rec_count = 0
    while True:
        bs = temp_file.read(8)
        if not bs:
            break
        sz = int.from_bytes(bs, "big", signed=False)
        rec = record_func()
        rec.ParseFromString(temp_file.read(sz))
        assert_func(rec)
        rec_count += 1
    assert num_records == rec_count


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
