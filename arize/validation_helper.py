import pandas as pd

from arize.input_transformer import _convert_element


def _validate_inputs(prediction_ids, prediction_labels, actual_labels,
                     features, model_id, model_version,
                     features_name_overwrite, timestamp_overwrite):
    if model_id is None:
        raise TypeError('model_id is NoneType, must be one of: str')
    if prediction_labels is None and actual_labels is None:
        raise TypeError(
            'either prediction_labels or actual_labels must be passed in, both are NoneType'
        )
    if prediction_ids is None:
        raise TypeError(
            'prediction_ids is None, but at least one prediction_id is required'
        )
    is_bulk = isinstance(prediction_ids, (pd.DataFrame, pd.Series))
    is_single = isinstance(prediction_ids, (str, bytes))
    if is_bulk:
        if prediction_labels is not None:
            _validate_bulk_prediction_inputs(prediction_ids, prediction_labels,
                                             features, features_name_overwrite,
                                             timestamp_overwrite)
        if actual_labels is not None:
            _validate_bulk_actuals_inputs(prediction_ids, actual_labels)
    elif is_single:
        if prediction_labels is not None:
            _validate_prediction_inputs(prediction_ids, prediction_labels,
                                        features, timestamp_overwrite)
        else:
            _validate_actual_inputs(prediction_ids, actual_labels)
    else:
        msg = f'prediction_ids is type {type(prediction_ids)}, but expect one of str, bytes, pd.DataFrame, pd.Series'
        raise TypeError(msg)
    return is_bulk


def _validate_bulk_prediction_inputs(prediction_ids, prediction_labels,
                                     features, features_name_overwrite,
                                     timestamp_overwrite):
    if timestamp_overwrite is not None:
        if isinstance(timestamp_overwrite, (pd.DataFrame, pd.Series)):
            if timestamp_overwrite.shape[0] != prediction_ids.shape[0]:
                msg = f'timestamp_overwrite has {timestamp_overwrite.shape[0]} but must have same number of elements as prediction_ids {prediction_ids.shape[0]}'
                raise ValueError(msg)
        elif isinstance(timestamp_overwrite, list):
            if prediction_ids.shape[0] != len(timestamp_overwrite):
                msg = f'timestamp_overwrite has length {len(timestamp_overwrite)} but must have same number of elements as prediction_ids {prediction_ids.shape[0]}'
                raise ValueError(msg)
        else:
            msg = f'timestamp_overwrite is type: {type(timestamp_overwrite)}, but expected: pd.DataFrame, pd.Series'
            raise TypeError(msg)
    if prediction_labels is None:
        raise ValueError(
            'at least one prediction label is required for prediction records')
    if not isinstance(prediction_labels, (pd.DataFrame, pd.Series)):
        msg = f'prediction_labels is type: {type(prediction_labels)}, but expects one of: pd.DataFrame, pd.Series'
        raise TypeError(msg)
    if prediction_labels.shape[0] != prediction_ids.shape[0]:
        msg = f'prediction_labels shaped {prediction_labels.shape} must have the same number of rows as predictions_ids shaped {prediction_ids.shape}.'
        raise ValueError(msg)
    if features is not None and not isinstance(features, pd.DataFrame):
        msg = f'features is type {type(features)}, but expect type pd.DataFrame.'
        raise TypeError(msg)
    if features is not None and features.shape[0] != prediction_ids.shape[0]:
        msg = f'features shaped {features.shape[0]} must have the same number of rows as predictions_ids shaped {prediction_ids.shape[0]}.'
        raise ValueError(msg)
    if features_name_overwrite is not None and features is not None and len(
            features.columns) != len(features_name_overwrite):
        msg = f'features_name_overwrite has len:{len(features_name_overwrite)}, but expects the same number of columns in features dataframe. ({len(features.columns)} columns).'
        raise ValueError(msg)
    if features is not None and isinstance(
            features.columns, pd.core.indexes.range.RangeIndex
    ) and features_name_overwrite is None:
        msg = f'fatures.columns is of type {type(features.columns)}, but expect  therefore, features_name_overwrite must be present to overwrite columns index with human readable feature names.'
        raise TypeError(msg)
    if features is not None and features.columns is not None:
        for feature in features.columns:
            if not isinstance(feature, str):
                msg = f'features.column {feature} is type {type(feature)}, but expect str'
                raise TypeError(msg)


def _validate_bulk_actuals_inputs(prediction_ids, actual_labels):
    if actual_labels is None:
        raise ValueError('at least one actual label is required')
    if not isinstance(actual_labels, (pd.DataFrame, pd.Series)):
        msg = f'actual_labels is type: {type(actual_labels)}, but expects one of: pd.DataFrame, pd.Series'
        raise TypeError(msg)
    if actual_labels.shape[0] != prediction_ids.shape[0]:
        msg = f'actual_labels shaped {actual_labels.shape[0]} must have the same number of rows as predictions_ids shaped {prediction_ids.shape[0]}.'
        raise ValueError(msg)


def _validate_prediction_inputs(prediction_ids, prediction_labels, features,
                                timestamp_overwrite):
    if prediction_labels is None:
        raise ValueError('at least one prediction label is required')
    if not isinstance(_convert_element(prediction_labels),
                      (str, bool, float, int)):
        msg = f'prediction_label {prediction_labels} has type {type(prediction_labels)}, but expected one of: str, bool, float, int'
        raise TypeError(msg)
    if features is not None and bool(features):
        for k, v in features.items():
            if not isinstance(_convert_element(v), (str, bool, float, int)):
                msg = f'feature {k} with value {v} is type {type(v)}, but expected one of: str, bool, float, int'
                raise TypeError(msg)
    if timestamp_overwrite is not None and not isinstance(
            timestamp_overwrite, int):
        msg = f'timestamp_overwrite {timestamp_overwrite} is type {type(timestamp_overwrite)} but expected int'
        raise TypeError(msg)


def _validate_actual_inputs(prediction_ids, actual_labels):
    if actual_labels is None:
        raise ValueError('at least one actual label is required')
    if not isinstance(_convert_element(actual_labels), (str, bool, float, int)):
        msg = f'actuals_labels {actual_labels} has type {type(actual_labels)}, but expected one of: str, bool, float, int'
        raise ValueError(msg)
