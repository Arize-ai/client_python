import datetime
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import arize.public_pb2 as public__pb2
from arize.model import TrainingRecords, ValidationRecords
from arize.types import ModelTypes
from arize.api import Client

NUM_VAL = 20.20
STR_VAL = 'arize'
BOOL_VAL = True
INT_VAL = 0
NP_FLOAT = np.float(1.2)
file_to_open = Path(__file__).parent / "fixtures/mpg.csv"

expected = {
    'model': 'model_v0',
    'model_version': 'v1.2.3.4',
    'batch': 'batch1234',
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
        'feature_bool': BOOL_VAL,
        'feature_None': None
    },
    'feature_importances': {
        'feature_str': NUM_VAL,
        'feature_double': NUM_VAL,
        'feature_int': NUM_VAL,
        'feature_bool': NUM_VAL,
        'feature_numpy_float': NP_FLOAT,
    }
}


def mock_dataframes_clean_nan(file):
    features, labels, ids = mock_dataframes(file)
    features = features.fillna('backfill')
    return features, labels, ids


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

def get_stubbed_client():
    c = Client(organization_key="test_org", api_key="API_KEY", uri="https://localhost:443")

    def _post(record, uri, indexes):
        return record

    def _post_bulk(records, uri):
        return records

    def _post_preprod(records):
        return records

    c._post = _post
    c._post_bulk = _post_bulk
    c._post_preprod = _post_preprod
    return c

# TODO for each existing test that has been modified to call Client.log, add a call
# to the pre-existing method that should map to the identical cacll to Client.log to
# assert that they are equivalent


def test_build_binary_prediction_features():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                   model_type=ModelTypes.BINARY,
                   model_version=expected['model_version'],
                   prediction_id=expected['prediction_id'],
                   prediction_label=expected['value_binary'],
                   features=expected['features'],
                   prediction_timestamp=None)

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
    assert record.prediction.timestamp.seconds == 0
    assert record.prediction.timestamp.nanos == 0

def test_build_binary_prediction_features():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                   model_type=ModelTypes.BINARY,
                   model_version=expected['model_version'],
                   prediction_id=expected['prediction_id'],
                   prediction_label=expected['value_binary'],
                   features=expected['features'],
                   prediction_timestamp=None)

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
    assert record.prediction.timestamp.seconds == 0
    assert record.prediction.timestamp.nanos == 0


def test_build_binary_prediction_zero_ones():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                   model_type=ModelTypes.BINARY,
                   model_version=expected['model_version'],
                   prediction_id=expected['prediction_id'],
                   prediction_label=1,
                   features=expected['features'],
                   prediction_timestamp=None)

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
    assert record.prediction.timestamp.seconds == 0
    assert record.prediction.timestamp.nanos == 0


def test_build_categorical_prediction():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                   model_version=expected['model_version'],
                   prediction_id=expected['prediction_id'],
                   prediction_label=expected['value_categorical'],
                   features=expected['features'],
                   prediction_timestamp=None)
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert isinstance(record.prediction.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert bool(record.prediction.features)
    assert record.prediction.label.categorical == expected['value_categorical']


def test_build_scored_prediction():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                   model_type=ModelTypes.SCORE_CATEGORICAL,
                   model_version=expected['model_version'],
                   prediction_id=expected['prediction_id'],
                   prediction_label=(expected['value_categorical'], expected['value_numeric']),
                   features=expected['features'],
                   prediction_timestamp=None)
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert isinstance(record.prediction.label, public__pb2.Label)
    assert isinstance(record.prediction.label.score_categorical, public__pb2.ScoreCategorical)

    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert bool(record.prediction.features)
    assert record.prediction.label.score_categorical.score == expected['value_numeric']
    assert record.prediction.label.score_categorical.categorical == expected['value_categorical']


def test_build_scored_actual():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                  model_type=ModelTypes.SCORE_CATEGORICAL,
                  prediction_id=expected['prediction_id'],
                  actual_label=expected['value_categorical'])
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert isinstance(record.actual.label.score_categorical, public__pb2.ScoreCategorical)

    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    # 0.0 is the default float representation
    assert record.actual.label.score_categorical.score == 0.0
    assert record.actual.label.score_categorical.categorical == expected['value_categorical']


def test_build_numeric_prediction():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_numeric'],
                      features=expected['features'],
                      prediction_timestamp=None)
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
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_numeric'],
                      features=None,
                      prediction_timestamp=None)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert not bool(record.prediction.features)


def test_build_numeric_actual():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_numeric'])
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.numeric == expected['value_numeric']
    assert record.actual.timestamp.seconds == 0
    assert record.actual.timestamp.nanos == 0


def test_build_categorical_actual():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_categorical'])
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.categorical == expected['value_categorical']


def test_build_binary_actual():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_binary'])
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.binary == expected['value_binary']


######################
## Bulk Log Tests ####
######################

def test_build_bulk_predictions_dataframes():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    bulk_records = c.bulk_log(model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=None,
                           prediction_timestamps=None)
    record_count = 0
    for indexes, bulk in bulk_records.items():
        assert indexes == (0, len(ids))
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert bulk.model_version == expected['model_version']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'numeric'
            assert record.prediction.timestamp.seconds == 0
            assert record.prediction.timestamp.nanos == 0
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_scored_predictions():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    scores = pd.DataFrame(data=np.random.random(size=(features.shape[0], 1)))
    labels.columns = ["category"]
    scores.columns = ["score"]
    labels = labels.astype(str)
    score_labels = pd.concat([labels, scores], axis=1)
    bulk_records = c.bulk_log(model_id=expected['model'],
                           model_type=ModelTypes.SCORE_CATEGORICAL,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=score_labels,
                           features=features,
                           feature_names_overwrite=None,
                           prediction_timestamps=None)
    record_count = 0
    for indexes, bulk in bulk_records.items():
        assert indexes == (0, len(ids))
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert bulk.model_version == expected['model_version']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert isinstance(record.prediction.label.score_categorical, public__pb2.ScoreCategorical)
            assert len(record.prediction.features) == features.shape[1]
            assert isinstance(record.prediction.label.score_categorical.score, float)
            assert record.prediction.label.score_categorical.score == scores["score"][record_count]
            assert isinstance(record.prediction.label.score_categorical.categorical, str)
            assert record.prediction.timestamp.seconds == 0
            assert record.prediction.timestamp.nanos == 0
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_predictions_dataframes_with_nans():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes(file_to_open)
    features.horsepower = np.nan
    bulk_records = c.bulk_log(model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=None,
                           prediction_timestamps=None)
    record_count=0
    for indexes, bulk in bulk_records.items():
        assert indexes == (0, len(ids))
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert bulk.model_version == expected['model_version']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == (features.shape[1] - 1)
            assert record.prediction.label.WhichOneof('data') == 'numeric'
            assert record.prediction.timestamp.seconds == 0
            assert record.prediction.timestamp.nanos == 0
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_predictions_no_features():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    records = c.bulk_log(model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=None,
                           feature_names_overwrite=None,
                           prediction_timestamps=None)
    for _, bulk in records.items():
        assert isinstance(bulk, public__pb2.BulkRecord)
        for r in bulk.records:
            assert isinstance(r, public__pb2.Record)
            assert not bool(r.organization_key)
            assert not bool(r.model_id)
            assert not bool(r.prediction.features)


def test_build_bulk_prediction_with_feature_names_overwrites():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    feature_names_overwrite = [
        'mask_' + str(i) for i in range(len(features.columns))
    ]
    records = c.bulk_log(model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=feature_names_overwrite,
                           prediction_timestamps=None)
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
    c = get_stubbed_client()
    _, labels, ids = mock_dataframes_clean_nan(file_to_open)
    bulk_records = c.bulk_log(model_id=expected['model'],
                         prediction_ids=ids,
                         actual_labels=labels)
    record_count = 0
    for indexes, bulk in bulk_records.items():
        assert indexes == (0, len(ids))
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.actual.label, public__pb2.Label)
            assert record.prediction_id == ids[0][record_count]
            assert record.actual.label.WhichOneof('data') == 'numeric'
            assert record.actual.timestamp.seconds == 0
            assert record.actual.timestamp.nanos == 0
            record_count += 1
    assert record_count == len(ids)


def test_validate_bulk_predictions_timestamp_out_of_range():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)

    current_time = datetime.datetime.now().timestamp()
    earlier_time = (datetime.datetime.now() - datetime.timedelta(days=30)).timestamp()
    prediction_timestamps = np.linspace(earlier_time, current_time, num=len(ids))
    prediction_timestamps = pd.Series(prediction_timestamps.astype(int))

    # break one of the timestamps
    prediction_timestamps.iloc[4] = int(current_time) + (366 * 24 * 60 * 60)
    ex = None
    try:
        c.bulk_log(
            model_id=expected['model'],
            model_version=expected['model_version'],
            prediction_ids=ids,
            prediction_labels=labels,
            features=features,
            prediction_timestamps=prediction_timestamps,
        )
    except Exception as err:
        ex = err

    assert isinstance(ex, ValueError)


def test_validate_bulk_predictions_mismatched_shapes():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    feature_names_overwrite = [
        'mask_' + str(i) for i in range(len(features.columns))
    ]
    id_ex, feature_ex, label_ex, overwrite_ex = None, None, None, None
    try:
        c.bulk_log(model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids[3:],
                               prediction_labels=labels,
                               features=features,
                               feature_names_overwrite=feature_names_overwrite,
                               prediction_timestamps=None)
    except Exception as err:
        id_ex = err
    try:
        c.bulk_log(model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels,
                               features=features[3:],
                               feature_names_overwrite=None,
                               prediction_timestamps=None)
    except Exception as err:
        feature_ex = err
    try:
       c.bulk_log(model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels[3:],
                               features=None,
                               feature_names_overwrite=None,
                               prediction_timestamps=None)
    except Exception as err:
        label_ex = err
    try:
        c.bulk_log(model_id=expected['model'],
            model_version=expected['model_version'],
            prediction_ids=ids,
            prediction_labels=labels,
            features=features,
            feature_names_overwrite=feature_names_overwrite[3:],
            prediction_timestamps=None)
    except Exception as err:
        overwrite_ex = err
    assert isinstance(id_ex, ValueError)
    assert isinstance(feature_ex, ValueError)
    assert isinstance(label_ex, ValueError)
    assert isinstance(overwrite_ex, ValueError)


def test_validate_bulk_predictions_default_columns_int():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    features_default_columns = pd.DataFrame(features[:].values)
    ex = None
    try:
        c.bulk_log(model_id=expected['model'],
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels,
                               features=features_default_columns,
                               feature_names_overwrite=None,
                               prediction_timestamps=None)
    except Exception as err:
        ex = err
    assert isinstance(ex, TypeError)


def test_build_bulk_prediction_with_prediction_timestamps():
    c = get_stubbed_client()
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    t = [int(time.time()) + i for i in range(features.shape[0])]
    records = c.bulk_log(model_id=expected['model'],
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           feature_names_overwrite=None,
                           prediction_timestamps=t)
    for _, bulk in records.items():
        assert isinstance(bulk, public__pb2.BulkRecord)
        for r in bulk.records:
            assert isinstance(r, public__pb2.Record)
            assert not bool(r.organization_key)
            assert not bool(r.model_id)
            assert bool(r.prediction.features)
            assert r.prediction.timestamp is not None


def test_handle_log_prediction_with_prediction_timestamps():
    t = int(time.time())
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                       model_version=expected['model_version'],
                       prediction_id=expected['prediction_id'],
                       prediction_label=expected['value_binary'],
                       features=expected['features'],
                       prediction_timestamp=t)
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert bool(record.prediction.features)
    assert record.prediction.timestamp.seconds == t


def test_build_bulk_predictions_index():
    c = get_stubbed_client()
    features, labels, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    bulk_records = c.bulk_log(model_id=expected['model'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           features=features,
                           model_version=expected['model_version'],
                           feature_names_overwrite=None,
                           prediction_timestamps=None)
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'numeric'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_actuals_index():
    c = get_stubbed_client()
    _, labels, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    bulk_records = c.bulk_log(model_id=expected['model'],
                         prediction_ids=ids,
                         actual_labels=labels)
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.actual.label, public__pb2.Label)
            assert record.prediction_id == ids[record_count][0]
            assert record.actual.label.WhichOneof('data') == 'numeric'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_binary_predictions():
    c = get_stubbed_client()
    features, _, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    features['pred'] = features['mpg'].apply(lambda x: x > 15)
    bulk_records = c.bulk_log(model_id=expected['model'],
                           prediction_ids=ids,
                           prediction_labels=features['pred'],
                           features=features,
                           model_version=expected['model_version'],
                           feature_names_overwrite=None,
                           prediction_timestamps=None)
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'binary'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_binary_actuals():
    c = get_stubbed_client()
    features, _, idx = mock_dataframes_clean_nan(file_to_open)
    features['actual'] = features['mpg'].apply(lambda x: x > 15)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    bulk_records = c.bulk_log(model_id=expected['model'],
                         prediction_ids=ids,
                         actual_labels=features['actual'])
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.actual.label, public__pb2.Label)
            assert record.prediction_id == ids[record_count][0]
            assert record.actual.label.WhichOneof('data') == 'binary'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_build_feature_importances():
    c = get_stubbed_client()
    record = c.log(model_id=expected['model'],
                   prediction_id=expected['prediction_id'],
                   shap_values=expected['feature_importances'])
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.feature_importances, public__pb2.FeatureImportances)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert len(record.feature_importances.feature_importances) == len(expected['feature_importances'])


def test_prediction_timestamp_out_of_range():
    c = get_stubbed_client()
    ex = None

    try:
        c.log(
            model_id=expected['model'],
            prediction_id=expected['prediction_id'],
            model_version=expected['model_version'],
            model_type=ModelTypes.CATEGORICAL,
            prediction_label='HOTDOG',
            features=expected['features'],
            prediction_timestamp=int(time.time()) + (380 * 24 * 60 * 60),
        )
    except Exception as err:
        ex = err

    assert isinstance(ex, ValueError)


def test_build_missing_data():
    c = get_stubbed_client()
    ex = None

    try:
        c.log(model_id=expected['model'],
              prediction_id=expected['prediction_id'])
    except Exception as err:
        # Error because everything is None
        ex = err

    assert isinstance(ex, ValueError)


def test_build_feature_importances_error_empty_data():
    c = get_stubbed_client()
    ex = None

    try:
        c.log(model_id=expected['model'],
                                prediction_id=expected['prediction_id'],
                                shap_values={}
                                )
    except Exception as err:
        # Error because no feature_importances were provided
        ex = err

    assert isinstance(ex, ValueError)


def test_build_feature_importances_error_wrong_data_type():
    c = get_stubbed_client()
    ex = None
    try:
        c.log(model_id=expected['model'],
                                prediction_id=expected['prediction_id'],
                                shap_values={"a": "string"}
                                # feature importances should be float, so this will produce an error
                                )
    except Exception as err:
        ex = err

    assert isinstance(ex, TypeError)


def test_build_bulk_feature_importances():
    c = get_stubbed_client()
    features, _, pred_ids = mock_dataframes_clean_nan(file_to_open)

    data = np.random.rand(len(pred_ids), len(features.columns))
    feature_importances = pd.DataFrame(data=data, columns=features.columns)
    ids = pd.DataFrame(index=pred_ids.values, data=pred_ids.values).index.to_series()

    bulk_records = c.bulk_log(model_id=expected['model'],
                                      prediction_ids=ids,
                                      shap_values=feature_importances)
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            fi = record.feature_importances
            assert isinstance(fi, public__pb2.FeatureImportances)
            assert len(fi.feature_importances) == len(features.columns)

            assert record.prediction_id == ids[record_count][0]
            assert record.prediction_id in pred_ids.values
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_feature_importances_error_mismatch():
    c = get_stubbed_client()
    features, _, pred_ids = mock_dataframes_clean_nan(file_to_open)

    # Make the length of feature importances data array mismatch the number of prediction ids
    data = np.random.rand(len(pred_ids) - 1, len(features.columns))

    feature_importances = pd.DataFrame(data=data, columns=features.columns)
    ids = pd.DataFrame(index=pred_ids.values, data=pred_ids.values).index.to_series()

    ex = None
    try:
        c.bulk_log(model_id=expected['model'],
                   prediction_ids=ids,
                   shap_values=feature_importances)
    except Exception as err:
        # feature importances data length and number of prediction ids mismatch should cause this error
        ex = err

    assert isinstance(ex, ValueError)


# def test_build_bulk_feature_importances_error_wrong_data_type():
#     features, _, pred_ids = mock_dataframes(file_to_open)
#
#     # Replace one of the rows in the feature importances data with values of the wrong data type (i.e. not float)
#     data = np.random.rand(len(pred_ids) - 1, len(features.columns))
#     data_wrong_type = np.ones(len(features.columns), dtype=bool)
#
#     data = np.vstack((data, data_wrong_type))
#     feature_importances = pd.DataFrame(data=data, columns=features.columns)
#     ids = pd.DataFrame(index=pred_ids.values, data=pred_ids.values).index.to_series()
#
#     ex = None
#     try:
#         bulk_fi = BulkFeatureImportances(organization_key=expected['organization_key'],
#                                          model_id=expected['model'],
#                                          prediction_ids=ids,
#                                          feature_importances=feature_importances)
#
#         bulk_fi.validate_inputs()
#     except Exception as err:
#         # caused by wrong type
#         ex = err
#
#     assert isinstance(ex, ValueError)


def test_build_training_records():
    features, labels, _ = mock_dataframes_clean_nan(file_to_open)
    recs = TrainingRecords(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           model_version=expected['model_version'],
                           prediction_labels=labels,
                           actual_labels=labels,
                           features=features)
    bundles = recs.build_proto()
    record_count = 0
    for _, recs in bundles.items():
        for rec in recs:
            record_count += 1
            assert isinstance(rec, public__pb2.PreProductionRecord)
            assert isinstance(rec.training_record, public__pb2.PreProductionRecord.TrainingRecord)
            assert isinstance(rec.training_record.record, public__pb2.Record)
            assert rec.training_record.record.organization_key == expected['organization_key']
            assert rec.training_record.record.model_id == expected['model']
            assert rec.training_record.record.prediction_and_actual.prediction.model_version == expected['model_version']
            assert isinstance(rec.training_record.record.prediction_and_actual.prediction.label, public__pb2.Label)
            assert len(rec.training_record.record.prediction_and_actual.prediction.features) == features.shape[1]
            assert rec.training_record.record.prediction_and_actual.prediction.label.WhichOneof('data') == 'numeric'
            assert rec.training_record.record.prediction_and_actual.prediction.timestamp.seconds == 0
            assert rec.training_record.record.prediction_and_actual.prediction.timestamp.nanos == 0
    assert record_count == len(labels)


def test_send_validation_records():
    c = get_stubbed_client()
    features, labels, pred_ids = mock_dataframes_clean_nan(file_to_open)
    t = [int(time.time()) + i for i in range(features.shape[0])]

    # make life a bit easier and just take the first record
    features = features[:1]
    labels = labels[:1]
    pred_ids = pred_ids[:1]
    t = t[:1]

    result = c.log_validation_records(
        model_id=expected['model'],
        model_version=expected['model_version'],
        batch_id=expected['batch'],
        prediction_labels=labels,
        actual_labels=labels,
        prediction_ids=pred_ids,
        model_type=ModelTypes.NUMERIC,
        features=features,
        prediction_timestamps=t,
    )

    # test values in single record
    expected_prediction_id = pred_ids[0][0]
    for _, recs in result.items():
        for rec in recs:
            assert isinstance(rec, public__pb2.PreProductionRecord)
            assert isinstance(rec.validation_record, public__pb2.PreProductionRecord.ValidationRecord)
            assert isinstance(rec.validation_record.record, public__pb2.Record)
            assert rec.validation_record.batch_id == expected['batch']
            assert rec.validation_record.record.organization_key == expected['organization_key']
            assert rec.validation_record.record.model_id == expected['model']
            assert rec.validation_record.record.prediction_and_actual.prediction.model_version == expected['model_version']
            assert isinstance(rec.validation_record.record.prediction_and_actual.prediction.label, public__pb2.Label)
            assert len(rec.validation_record.record.prediction_and_actual.prediction.features) == features.shape[1]
            assert rec.validation_record.record.prediction_and_actual.prediction.label.WhichOneof('data') == 'numeric'
            assert rec.validation_record.record.prediction_and_actual.prediction.timestamp.seconds == t[0]
            assert rec.validation_record.record.prediction_and_actual.prediction.timestamp.nanos == 0
            assert rec.validation_record.record.prediction_id == expected_prediction_id

    # now test a bunch of records at once
    features, labels, pred_ids = mock_dataframes_clean_nan(file_to_open)
    result = c.log_validation_records(
        model_id=expected['model'],
        model_version=expected['model_version'],
        batch_id=expected['batch'],
        prediction_labels=labels,
        actual_labels=labels,
        prediction_ids=pred_ids,
        model_type=ModelTypes.NUMERIC,
        features=features,
    )
    records_count = 0
    for _, recs in result.items():
        for _ in recs:
            records_count += 1
    assert len(labels) == records_count


def test_send_validation_records_without_prediction_id():
    c = get_stubbed_client()
    features, labels, pred_ids = mock_dataframes_clean_nan(file_to_open)
    # expect no exceptions
    c.log_validation_records(
        model_id=expected['model'],
        model_version=expected['model_version'],
        batch_id=expected['batch'],
        prediction_labels=labels,
        actual_labels=labels,
        model_type=ModelTypes.NUMERIC,
        features=features,
    )


def test_build_bulk_binary_predictions_deprecated_method():
    c = get_stubbed_client()
    features, _, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    features['pred'] = features['mpg'].apply(lambda x: x > 15)
    bulk_records = c.log_bulk_predictions(model_id=expected['model'],
                           prediction_ids=ids,
                           prediction_labels=features['pred'],
                           features=features,
                           model_version=expected['model_version'])
    record_count = 0
    for _, bulk in bulk_records.items():
        assert bulk.organization_key == expected['organization_key']
        assert bulk.model_id == expected['model']
        assert not hasattr(bulk, 'timestamp')
        for record in bulk.records:
            assert isinstance(record, public__pb2.Record)
            assert isinstance(record.prediction.label, public__pb2.Label)
            assert len(record.prediction.features) == features.shape[1]
            assert record.prediction.label.WhichOneof('data') == 'binary'
            assert record.prediction_id in idx.values
            record_count += 1
    assert record_count == len(ids)


def test_validation_predictions_ids_as_index_series():
    c = get_stubbed_client()
    features, labels, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    result = c.log_validation_records(
        model_id=expected['model'],
        model_version=expected['model_version'],
        batch_id=expected['batch'],
        prediction_labels=labels,
        actual_labels=labels,
        prediction_ids=ids,
        features=features,
    )

    record_count = 0
    for _, recs in result.items():
        for rec in recs:
            record_count += 1
            assert isinstance(rec, public__pb2.PreProductionRecord)
            assert isinstance(rec.validation_record, public__pb2.PreProductionRecord.ValidationRecord)
            assert isinstance(rec.validation_record.record, public__pb2.Record)
            assert rec.validation_record.batch_id == expected['batch']
            assert rec.validation_record.record.organization_key == expected['organization_key']
            assert rec.validation_record.record.model_id == expected['model']
            assert rec.validation_record.record.prediction_and_actual.prediction.model_version == expected['model_version']
            assert isinstance(rec.validation_record.record.prediction_and_actual.prediction.label, public__pb2.Label)
            assert len(rec.validation_record.record.prediction_and_actual.prediction.features) == features.shape[1]
            assert rec.validation_record.record.prediction_and_actual.prediction.label.WhichOneof('data') == 'numeric'
            assert rec.validation_record.record.prediction_id in idx.values
    assert len(labels) == record_count


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__]))
