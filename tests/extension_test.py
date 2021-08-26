import uuid
import pandas as pd
import numpy as np
from arize import public_pb2 as pb
from arize.utils.types import ModelTypes, Environments
from arize.pandas.logger import Client, Schema
import tempfile

class TestClient(Client):
    def _post_file(self, path):
        pass

def test_production_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    client.log(
        df,
        temp_file.name,
        "model_id",
        "model_version",
        None, ModelTypes.NUMERIC,
        Environments.PRODUCTION,
        Schema(
            prediction_id_column_name="prediction_id",
            feature_column_names=list("ABCDEFGHIJKL"),
            prediction_label_column_name="prediction_label",
            actual_label_column_name="actual_label",
            shap_values_column_names={'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}))

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.PRODUCTION == header.environment

    rec_count = 0
    while True:
        bs = temp_file.read(8)
        if not bs:
            break
        sz = int.from_bytes(bs, "big", signed=False)
        rec = pb.Record()
        rec.ParseFromString(temp_file.read(sz))
        assert rec.prediction_id is not None
        assert rec.prediction.label is not None
        assert rec.actual.label is not None
        assert rec.prediction.features is not None
        assert len(rec.prediction.features) == 12
        assert len(rec.feature_importances.feature_importances) == 4
        rec_count += 1
    assert num_records == rec_count


def test_training_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    client.log(
        df,
        temp_file.name,
        "model_id",
        "model_version",
        None, ModelTypes.NUMERIC,
        Environments.TRAINING,
        Schema(prediction_id_column_name="prediction_id",
               feature_column_names=list("ABCDEFGHIJKL"),
               prediction_label_column_name="prediction_label",
               actual_label_column_name="actual_label"))

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.TRAINING == header.environment

    rec_count = 0
    while True:
        bs = temp_file.read(8)
        if not bs:
            break
        sz = int.from_bytes(bs, "big", signed=False)
        rec = pb.PreProductionRecord()
        rec.ParseFromString(temp_file.read(sz))
        assert rec.training_record.record.prediction_id is not None
        assert rec.training_record.record.prediction.label is not None
        assert rec.training_record.record.actual.label is not None
        assert rec.training_record.record.prediction.features is not None
        assert len(rec.training_record.record.prediction.features) == 12
        rec_count += 1
    assert num_records == rec_count


def test_validation_roundtrip():
    client = TestClient("apikey", "organizationkey")
    num_records = 100
    temp_file = tempfile.NamedTemporaryFile()
    df = build_df(num_records)
    client.log(
        df,
        temp_file.name,
        "model_id",
        "model_version",
        "batch_id", ModelTypes.NUMERIC,
        Environments.VALIDATION,
        Schema(prediction_id_column_name="prediction_id",
               feature_column_names=list("ABCDEFGHIJKL"),
               prediction_label_column_name="prediction_label",
               actual_label_column_name="actual_label"))

    header = get_header(temp_file)
    assert pb.FileHeader.Environment.VALIDATION == header.environment

    rec_count = 0
    while True:
        bs = temp_file.read(8)
        if not bs:
            break
        sz = int.from_bytes(bs, "big", signed=False)
        rec = pb.PreProductionRecord()
        rec.ParseFromString(temp_file.read(sz))
        assert rec.validation_record.batch_id == "batch_id"
        assert rec.validation_record.record.prediction_id is not None
        assert rec.validation_record.record.prediction.label is not None
        assert rec.validation_record.record.actual.label is not None
        assert rec.validation_record.record.prediction.features is not None
        assert len(rec.validation_record.record.prediction.features) == 12
        rec_count += 1
    assert num_records == rec_count


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
    pred_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(num_records, 1)), columns=["prediction_label"])
    actuals_labels = pd.DataFrame(np.random.randint(0, 100000000, size=(num_records, 1)), columns=["actual_label"])
    fi = pd.DataFrame(
        np.random.randint(0, 100000000, size=(num_records, 12)),
        columns=list("abcdefghijkl"),
    )

    ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(num_records)], columns=["prediction_id"])
    return pd.concat([features, pred_labels, actuals_labels, ids, fi], axis=1)
