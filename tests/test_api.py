import pandas as pd
import numpy as np
import uuid
from pathlib import Path

from google.protobuf.timestamp_pb2 import Timestamp

import arize.public_pb2 as public__pb2
from arize.model import Prediction, Actual, BulkPrediction, BulkActual

NUM_VAL = 20.20
STR_VAL = 'arize'
BOOL_VAL = True
INT_VAL = 0
file_to_open = Path(__file__).parent / "fixtures/mpg.csv"

expected = {
    'model': 'model_v0',
    'model_version': 'v1.2.3.4',
    'api_key': 'API_KEY',
    'prediction_id': 'prediction_0',
    'value_binary': BOOL_VAL,
    'value_categorical': STR_VAL,
    'value_numeric': NUM_VAL,
    'organization_key': 'test_org',
    'features': {
        'feature_str': STR_VAL,
        'feature_double': NUM_VAL,
        'feature_int': INT_VAL,
        'feature_bool': BOOL_VAL
    }
}


def mock_dataframes(file):
    features = pd.read_csv(file)
    labels = pd.DataFrame(
        np.random.randint(1, 100, size=(features.shape[0], 1)))
    ids = pd.DataFrame([str(uuid.uuid4()) for _ in range(len(labels.index))])
    return features, labels, ids


def mock_series(file):
    features = pd.read_csv(file)
    labels = pd.Series(np.random.randint(1, 100, size=features.shape[0]))
    ids = pd.Series([str(uuid.uuid4()) for _ in range(len(labels.index))])
    return features, labels, ids


def test_build_binary_prediction_features():
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_binary'],
                      features=expected['features'],
                      time_overwrite=None)

    record = pred._build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert isinstance(record.prediction.label, public__pb2.Label)
    for feature in record.prediction.features:
        assert isinstance(record.prediction.features[feature],
                          public__pb2.Value)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert record.prediction.label.binary == expected['value_binary']
    assert record.prediction.features['feature_str'].WhichOneof(
        'data') == 'string'
    assert record.prediction.features['feature_double'].WhichOneof(
        'data') == 'double'
    assert record.prediction.features['feature_int'].WhichOneof(
        'data') == 'int'
    assert record.prediction.features['feature_bool'].WhichOneof(
        'data') == 'string'


def test_build_categorical_prediction():
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_categorical'],
                      features=expected['features'],
                      time_overwrite=None)
    record = pred._build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert isinstance(record.prediction.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert bool(record.prediction.features)
    assert record.prediction.label.categorical == expected['value_categorical']


def test_build_numeric_prediction():
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_numeric'],
                      features=expected['features'],
                      time_overwrite=None)
    record = pred._build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert isinstance(record.prediction.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert bool(record.prediction.features)
    assert record.prediction.label.numeric == expected['value_numeric']


def test_build_prediction_no_features():
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_numeric'],
                      features=None,
                      time_overwrite=None)
    record = pred._build_proto()
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert not bool(record.prediction.features)


def test_build_numeric_actual():
    actual = Actual(organization_key=expected['organization_key'],
                    model_id=expected['model'],
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_numeric'])
    record = actual._build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.numeric == expected['value_numeric']


def test_build_categorical_actual():
    actual = Actual(organization_key=expected['organization_key'],
                    model_id=expected['model'],
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_categorical'])
    record = actual._build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.categorical == expected['value_categorical']


def test_build_binary_actual():
    actual = Actual(organization_key=expected['organization_key'],
                    model_id=expected['model'],
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_binary'])
    record = actual._build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.binary == expected['value_binary']


def test_build_bulk_predictions_dataframes():
    features, labels, ids = mock_dataframes(file_to_open)
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds._build_proto()
    record_count = 0
    for indexes, bulk in bulk_records.items():
        assert indexes == (0, len(ids))
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert bulk.model_version == expected['model_version']
        assert isinstance(bulk.timestamp, Timestamp)
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'numeric'
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_predictions_no_features():
    features, labels, ids = mock_dataframes(file_to_open)
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=None,
                           feature_names_overwrite=None,
                           time_overwrite=None)
    records = preds._build_proto()
    for _, bulk in records.items():
        assert isinstance(bulk, public__pb2.BulkRecord)
        for r in bulk.records:
            assert isinstance(r, public__pb2.Record)
            assert not bool(r.organization_key)
            assert not bool(r.model_id)
            assert not bool(r.prediction.features)


def test_build_bulk_prediction_with_feature_names_overwrites():
    features, labels, ids = mock_dataframes(file_to_open)
    feature_names_overwrite = [
        'mask_' + str(i) for i in range(len(features.columns))
    ]
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=feature_names_overwrite,
                           time_overwrite=None)
    records = preds._build_proto()
    for _, bulk in records.items():
        assert isinstance(bulk, public__pb2.BulkRecord)
        for r in bulk.records:
            assert isinstance(r, public__pb2.Record)
            assert not bool(r.organization_key)
            assert not bool(r.model_id)
            assert bool(r.prediction.features)
            for feature in r.prediction.features:
                assert feature in feature_names_overwrite


def test_build_bulk_actuals_dataframes():
    _, labels, ids = mock_dataframes(file_to_open)
    actuals = BulkActual(organization_key=expected['organization_key'],
                         model_id=expected['model'],
                         prediction_ids=ids,
                         actual_labels=labels)
    bulk_records = actuals._build_proto()
    record_count = 0
    for indexes, bulk in bulk_records.items():
        assert indexes == (0, len(ids))
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert isinstance(bulk.timestamp, Timestamp)
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.actual.label, public__pb2.Label)
            assert record.prediction_id == ids[0][record_count]
            assert record.actual.label.WhichOneof('data') == 'numeric'
            record_count += 1
    assert record_count == len(ids)


def test_validate_bulk_predictions_mismatched_shapes():
    features, labels, ids = mock_dataframes(file_to_open)
    feature_names_overwrite = [
        'mask_' + str(i) for i in range(len(features.columns))
    ]
    id_ex, feature_ex, label_ex, overwrite_ex = None, None, None, None
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids[3:],
                               prediction_labels=labels,
                               features=features,
                               feature_names_overwrite=feature_names_overwrite,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        id_ex = err
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels,
                               features=features[3:],
                               feature_names_overwrite=None,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        feature_ex = err
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels[3:],
                               features=None,
                               feature_names_overwrite=None,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        label_ex = err
    try:
        preds = BulkPrediction(
            organization_key=expected['organization_key'],
            model_id=expected['model'],
            model_version=expected['model_version'],
            prediction_ids=ids,
            prediction_labels=labels,
            features=features,
            feature_names_overwrite=feature_names_overwrite[3:],
            time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        overwrite_ex = err
    assert isinstance(id_ex, ValueError)
    assert isinstance(feature_ex, ValueError)
    assert isinstance(label_ex, ValueError)
    assert isinstance(overwrite_ex, ValueError)


def test_validate_bulk_predictions_default_columns_int():
    features, labels, ids = mock_dataframes(file_to_open)
    features_default_columns = pd.DataFrame(features[:].values)
    ex = None
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels,
                               features=features_default_columns,
                               feature_names_overwrite=None,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        ex = err
    assert isinstance(ex, TypeError)


def test_build_bulk_prediction_with_time_overwrites():
    features, labels, ids = mock_dataframes(file_to_open)
    time = [1593626247 + i for i in range(features.shape[0])]
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=None,
                           time_overwrite=time)
    records = preds._build_proto()
    for _, bulk in records.items():
        assert isinstance(bulk, public__pb2.BulkRecord)
        for r in bulk.records:
            assert isinstance(r, public__pb2.Record)
            assert not bool(r.organization_key)
            assert not bool(r.model_id)
            assert bool(r.prediction.features)
            assert r.prediction.timestamp is not None


def test_handle_log_prediction_with_time_overwrites():
    preds = Prediction(organization_key=expected['organization_key'],
                       model_id=expected['model'],
                       model_version=expected['model_version'],
                       prediction_id=expected['prediction_id'],
                       prediction_label=expected['value_binary'],
                       features=expected['features'],
                       time_overwrite=1593626247)
    record = preds._build_proto()
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert bool(record.prediction.features)
    assert record.prediction.timestamp.seconds == 1593626247


def test_build_bulk_predictions_index():
    features, labels, idx = mock_dataframes(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           model_version=expected['model_version'],
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds._build_proto()
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert isinstance(bulk.timestamp, Timestamp)
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'numeric'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_actuals_index():
    _, labels, idx = mock_dataframes(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    actuals = BulkActual(organization_key=expected['organization_key'],
                         model_id=expected['model'],
                         prediction_ids=ids,
                         actual_labels=labels)
    bulk_records = actuals._build_proto()
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert isinstance(bulk.timestamp, Timestamp)
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.actual.label, public__pb2.Label)
            assert record.prediction_id == ids[record_count][0]
            assert record.actual.label.WhichOneof('data') == 'numeric'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_predictions_index_bool():
    features, _, idx = mock_dataframes(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    features['pred'] = features['mpg'].apply(lambda x: x > 15)
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           prediction_ids=ids,
                           prediction_labels=features['pred'],
                           features=features,
                           model_version=expected['model_version'],
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds._build_proto()
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert isinstance(bulk.timestamp, Timestamp)
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'binary'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_actuals_index_bool():
    features, _, idx = mock_dataframes(file_to_open)
    features['actual'] = features['mpg'].apply(lambda x: x > 15)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    actuals = BulkActual(organization_key=expected['organization_key'],
                         model_id=expected['model'],
                         prediction_ids=ids,
                         actual_labels=features['actual'])
    bulk_records = actuals._build_proto()
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert isinstance(bulk.timestamp, Timestamp)
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.actual.label, public__pb2.Label)
            assert record.prediction_id == ids[record_count][0]
            assert record.actual.label.WhichOneof('data') == 'binary'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)
