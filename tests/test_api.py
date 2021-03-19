import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import arize.public_pb2 as public__pb2
from arize.model import Prediction, Actual, BulkPrediction, BulkActual, FeatureImportances, BulkFeatureImportances, \
    TrainingRecords, ValidationRecords
from arize.types import ModelTypes

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
        'feature_bool': BOOL_VAL
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


def test_build_binary_prediction_features():
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_type=ModelTypes.BINARY,
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_binary'],
                      prediction_score=None,
                      features=expected['features'],
                      time_overwrite=None)

    record = pred.build_proto()
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
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_type=ModelTypes.BINARY,
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=1,
                      prediction_score=None,
                      features=expected['features'],
                      time_overwrite=None)

    record = pred.build_proto()
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
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_type=ModelTypes.CATEGORICAL,
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_categorical'],
                      prediction_score=None,
                      features=expected['features'],
                      time_overwrite=None)
    record = pred.build_proto()
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
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_type=ModelTypes.SCORE_CATEGORICAL,
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_score=expected['value_numeric'],
                      prediction_label=expected['value_categorical'],
                      features=expected['features'],
                      time_overwrite=None)
    record = pred.build_proto()
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
    pred = Actual(organization_key=expected['organization_key'],
                  model_id=expected['model'],
                  model_type=ModelTypes.SCORE_CATEGORICAL,
                  prediction_id=expected['prediction_id'],
                  actual_label=expected['value_categorical'])
    record = pred.build_proto()
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
    pred = Prediction(organization_key=expected['organization_key'],
                      model_id=expected['model'],
                      model_type=ModelTypes.NUMERIC,
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_numeric'],
                      prediction_score=None,
                      features=expected['features'],
                      time_overwrite=None)
    record = pred.build_proto()
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
                      model_type=ModelTypes.NUMERIC,
                      model_version=expected['model_version'],
                      prediction_id=expected['prediction_id'],
                      prediction_label=expected['value_numeric'],
                      prediction_score=None,
                      features=None,
                      time_overwrite=None)
    record = pred.build_proto()
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert not bool(record.prediction.features)


def test_build_numeric_actual():
    actual = Actual(organization_key=expected['organization_key'],
                    model_id=expected['model'],
                    model_type=ModelTypes.NUMERIC,
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_numeric'])
    record = actual.build_proto()
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
    actual = Actual(organization_key=expected['organization_key'],
                    model_id=expected['model'],
                    model_type=ModelTypes.CATEGORICAL,
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_categorical'])
    record = actual.build_proto()
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
                    model_type=ModelTypes.BINARY,
                    prediction_id=expected['prediction_id'],
                    actual_label=expected['value_binary'])
    record = actual.build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.actual, public__pb2.Actual)
    assert isinstance(record.actual.label, public__pb2.Label)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.actual.label.binary == expected['value_binary']


def test_build_bulk_predictions_dataframes():
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=None,
                           features=features,
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds.build_proto()
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
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    scores = pd.DataFrame(data=np.random.random(size=(features.shape[0], 1)))
    labels = labels.astype(str)

    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.SCORE_CATEGORICAL,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=scores,
                           features=features,
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds.build_proto()
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
            assert isinstance(record.prediction.label.score_categorical.categorical, str)
            assert record.prediction.timestamp.seconds == 0
            assert record.prediction.timestamp.nanos == 0
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_predictions_dataframes_with_nans():
    features, labels, ids = mock_dataframes(file_to_open)
    features.horsepower = np.nan
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=None,
                           features=features,
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds.build_proto()
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
            assert len(record.prediction.features) == (features.shape[1] - 1)
            assert record.prediction.label.WhichOneof('data') == 'numeric'
            assert record.prediction.timestamp.seconds == 0
            assert record.prediction.timestamp.nanos == 0
            record_count += 1
    assert record_count == len(ids)


def test_build_bulk_predictions_no_features():
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=None,
                           features=None,
                           feature_names_overwrite=None,
                           time_overwrite=None)
    records = preds.build_proto()
    for _, bulk in records.items():
        assert isinstance(bulk, public__pb2.BulkRecord)
        for r in bulk.records:
            assert isinstance(r, public__pb2.Record)
            assert not bool(r.organization_key)
            assert not bool(r.model_id)
            assert not bool(r.prediction.features)


def test_build_bulk_prediction_with_feature_names_overwrites():
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    feature_names_overwrite = [
        'mask_' + str(i) for i in range(len(features.columns))
    ]
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=None,
                           features=features,
                           feature_names_overwrite=feature_names_overwrite,
                           time_overwrite=None)
    records = preds.build_proto()
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
    _, labels, ids = mock_dataframes_clean_nan(file_to_open)
    actuals = BulkActual(organization_key=expected['organization_key'],
                         model_id=expected['model'],
                         model_type=ModelTypes.NUMERIC,
                         prediction_ids=ids,
                         actual_labels=labels)
    bulk_records = actuals.build_proto()
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


def test_validate_bulk_predictions_mismatched_shapes():
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    feature_names_overwrite = [
        'mask_' + str(i) for i in range(len(features.columns))
    ]
    id_ex, feature_ex, label_ex, overwrite_ex = None, None, None, None
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_type=ModelTypes.NUMERIC,
                               model_version=expected['model_version'],
                               prediction_ids=ids[3:],
                               prediction_labels=labels,
                               prediction_scores=None,
                               features=features,
                               feature_names_overwrite=feature_names_overwrite,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        id_ex = err
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_type=ModelTypes.NUMERIC,
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels,
                               prediction_scores=None,
                               features=features[3:],
                               feature_names_overwrite=None,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        feature_ex = err
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_type=ModelTypes.NUMERIC,
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels[3:],
                               prediction_scores=None,
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
            model_type=ModelTypes.NUMERIC,
            model_version=expected['model_version'],
            prediction_ids=ids,
            prediction_labels=labels,
            prediction_scores=None,
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
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    features_default_columns = pd.DataFrame(features[:].values)
    ex = None
    try:
        preds = BulkPrediction(organization_key=expected['organization_key'],
                               model_id=expected['model'],
                               model_type=ModelTypes.NUMERIC,
                               model_version=expected['model_version'],
                               prediction_ids=ids,
                               prediction_labels=labels,
                               prediction_scores=None,
                               features=features_default_columns,
                               feature_names_overwrite=None,
                               time_overwrite=None)
        preds.validate_inputs()
    except Exception as err:
        ex = err
    assert isinstance(ex, TypeError)


def test_build_bulk_prediction_with_time_overwrites():
    features, labels, ids = mock_dataframes_clean_nan(file_to_open)
    time = [1593626247 + i for i in range(features.shape[0])]
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           model_version=expected['model_version'],
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=None,
                           features=features,
                           feature_names_overwrite=None,
                           time_overwrite=time)
    records = preds.build_proto()
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
                       model_type=ModelTypes.NUMERIC,
                       model_version=expected['model_version'],
                       prediction_id=expected['prediction_id'],
                       prediction_label=expected['value_binary'],
                       prediction_score=None,
                       features=expected['features'],
                       time_overwrite=1593626247)
    record = preds.build_proto()
    assert isinstance(record.prediction, public__pb2.Prediction)
    assert bool(record.prediction.features)
    assert record.prediction.timestamp.seconds == 1593626247


def test_build_bulk_predictions_index():
    features, labels, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.NUMERIC,
                           prediction_ids=ids,
                           prediction_labels=labels,
                           prediction_scores=None,
                           features=features,
                           model_version=expected['model_version'],
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds.build_proto()
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
    _, labels, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    actuals = BulkActual(organization_key=expected['organization_key'],
                         model_id=expected['model'],
                         model_type=ModelTypes.NUMERIC,
                         prediction_ids=ids,
                         actual_labels=labels)
    bulk_records = actuals.build_proto()
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


def test_build_bulk_predictions_index_bool():
    features, _, idx = mock_dataframes_clean_nan(file_to_open)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    features['pred'] = features['mpg'].apply(lambda x: x > 15)
    preds = BulkPrediction(organization_key=expected['organization_key'],
                           model_id=expected['model'],
                           model_type=ModelTypes.BINARY,
                           prediction_ids=ids,
                           prediction_labels=features['pred'],
                           prediction_scores=None,
                           features=features,
                           model_version=expected['model_version'],
                           feature_names_overwrite=None,
                           time_overwrite=None)
    bulk_records = preds.build_proto()
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


def test_build_bulk_actuals_index_bool():
    features, _, idx = mock_dataframes_clean_nan(file_to_open)
    features['actual'] = features['mpg'].apply(lambda x: x > 15)
    ids = pd.DataFrame(index=idx.values, data=idx.values).index.to_series()
    actuals = BulkActual(organization_key=expected['organization_key'],
                         model_id=expected['model'],
                         model_type=ModelTypes.BINARY,
                         prediction_ids=ids,
                         actual_labels=features['actual'])
    bulk_records = actuals.build_proto()
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
    fi = FeatureImportances(organization_key=expected['organization_key'],
                            model_id=expected['model'],
                            prediction_id=expected['prediction_id'],
                            feature_importances=expected['feature_importances']
                            )
    record = fi.build_proto()
    assert isinstance(record, public__pb2.Record)
    assert isinstance(record.feature_importances, public__pb2.FeatureImportances)
    assert record.organization_key == expected['organization_key']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert len(record.feature_importances.feature_importances) == len(expected['feature_importances'])


def test_build_feature_importances_error_missing_data():
    ex = None

    try:
        fi = FeatureImportances(organization_key=expected['organization_key'],
                                model_id=expected['model'],
                                prediction_id=expected['prediction_id'],
                                feature_importances=None
                                )
        fi.validate_inputs()
    except Exception as err:
        # Error because feature_importances is None
        ex = err

    assert isinstance(ex, ValueError)


def test_build_feature_importances_error_empty_data():
    ex = None

    try:
        fi = FeatureImportances(organization_key=expected['organization_key'],
                                model_id=expected['model'],
                                prediction_id=expected['prediction_id'],
                                feature_importances={}
                                )
        fi.validate_inputs()
    except Exception as err:
        # Error because no feature_importances were provided
        ex = err

    assert isinstance(ex, ValueError)


def test_build_feature_importances_error_wrong_data_type():
    ex = None

    try:
        fi = FeatureImportances(organization_key=expected['organization_key'],
                                model_id=expected['model'],
                                prediction_id=expected['prediction_id'],
                                feature_importances={"a": "string"}
                                # feature importances should be float, so this will produce an error
                                )
        fi.validate_inputs()
    except Exception as err:
        ex = err

    assert isinstance(ex, TypeError)


def test_build_bulk_feature_importances():
    features, _, pred_ids = mock_dataframes_clean_nan(file_to_open)

    data = np.random.rand(len(pred_ids), len(features.columns))
    feature_importances = pd.DataFrame(data=data, columns=features.columns)
    ids = pd.DataFrame(index=pred_ids.values, data=pred_ids.values).index.to_series()

    bulk_req = BulkFeatureImportances(organization_key=expected['organization_key'],
                                      model_id=expected['model'],
                                      prediction_ids=ids,
                                      feature_importances=feature_importances)
    bulk_proto = bulk_req.build_proto()
    record_count = 0
    for _, bulk in bulk_proto.items():
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
    features, _, pred_ids = mock_dataframes_clean_nan(file_to_open)

    # Make the length of feature importances data array mismatch the number of prediction ids
    data = np.random.rand(len(pred_ids) - 1, len(features.columns))

    feature_importances = pd.DataFrame(data=data, columns=features.columns)
    ids = pd.DataFrame(index=pred_ids.values, data=pred_ids.values).index.to_series()

    ex = None
    try:
        bulk_fi = BulkFeatureImportances(organization_key=expected['organization_key'],
                                         model_id=expected['model'],
                                         prediction_ids=ids,
                                         feature_importances=feature_importances)

        bulk_fi.validate_inputs()
    except Exception as err:
        # feature importances data length and number of prediction ids mismatch should cause this error
        ex = err

    assert isinstance(ex, ValueError)


def test_build_bulk_feature_importances_error_wrong_data_type():
    features, _, pred_ids = mock_dataframes(file_to_open)

    # Replace one of the rows in the feature importances data with values of the wrong data type (i.e. not float)
    data = np.random.rand(len(pred_ids) - 1, len(features.columns))
    data_wrong_type = np.ones(len(features.columns), dtype=bool)

    data = np.vstack((data, data_wrong_type))
    feature_importances = pd.DataFrame(data=data, columns=features.columns)
    ids = pd.DataFrame(index=pred_ids.values, data=pred_ids.values).index.to_series()

    ex = None
    try:
        bulk_fi = BulkFeatureImportances(organization_key=expected['organization_key'],
                                         model_id=expected['model'],
                                         prediction_ids=ids,
                                         feature_importances=feature_importances)

        bulk_fi.validate_inputs()
    except Exception as err:
        # caused by wrong type
        ex = err

    assert isinstance(ex, ValueError)


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
    for _, rec in bundles.items():
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


def test_build_validation_records():
    features, labels, _ = mock_dataframes_clean_nan(file_to_open)
    recs = ValidationRecords(organization_key=expected['organization_key'],
                             model_id=expected['model'],
                             model_type=ModelTypes.NUMERIC,
                             model_version=expected['model_version'],
                             batch_id=expected['batch'],
                             prediction_labels=labels,
                             actual_labels=labels,
                             features=features)
    bundles = recs.build_proto()
    record_count = 0
    for _, rec in bundles.items():
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
        assert rec.validation_record.record.prediction_and_actual.prediction.timestamp.seconds == 0
        assert rec.validation_record.record.prediction_and_actual.prediction.timestamp.nanos == 0
    assert record_count == len(labels)
