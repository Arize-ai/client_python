import pandas as pd

from google.protobuf.timestamp_pb2 import Timestamp

from arize import public_pb2 as public__pb2


# Converts inputs from Series or lists to DataFrames for consitent interations downstream.
# It also converts elements in the frame to Arize supported data types
def _normalize_inputs(prediction_ids, prediction_labels, actual_labels,
                      features, features_name_overwrite, time_overwrite):

    ids, pred_labels_df, actual_labels_df, features_df, time_df = None, None, None, None, None
    ids = prediction_ids.to_numpy()

    if isinstance(prediction_labels, pd.Series):
        prediction_labels = prediction_labels.to_frame()
    if isinstance(actual_labels, pd.Series):
        actual_labels = actual_labels.to_frame()
    if isinstance(time_overwrite, pd.Series):
        time_overwrite = time_overwrite.to_frame()
    if isinstance(time_overwrite, (pd.Series, list)):
        time_overwrite = pd.DataFrame(time_overwrite)

    if prediction_labels is not None:
        pred_labels_df = prediction_labels.applymap(
            lambda x: _transform_label(x)).to_numpy()
    if actual_labels is not None:
        actual_labels_df = actual_labels.applymap(
            lambda x: _transform_label(x)).to_numpy()
    if features is not None:
        if features_name_overwrite is not None:
            features.columns = features_name_overwrite
        features_df = _build_value_map(features).to_dict('records')

    if time_overwrite is not None:
        time_df = time_overwrite.applymap(lambda x: _convert_time(x))
    return ids, pred_labels_df, actual_labels_df, features_df, time_df


def _convert_time(time):
    time = _convert_element(time)
    if not isinstance(time, int):
        msg = f'Time overwrite value {time} is type {type(time)}, but expects int. (Unix epoch time in seconds)'
        raise ValueError(msg)
    ts = None
    if time is not None:
        ts = Timestamp()
        ts.FromSeconds(time)
    return ts


def _build_value_map(vals):
    formatted = None
    if isinstance(vals, dict):
        formatted = {k: _tranform_value(v, k) for (k, v) in vals.items()}
    elif isinstance(vals, pd.DataFrame):
        formatted = vals.apply(
            lambda y: y.apply(lambda x: _tranform_value(x, y.name)))
    return formatted


def _tranform_value(value, name):
    if isinstance(value, public__pb2.Value):
        return value
    val = _convert_element(value)
    if isinstance(val, (str, bool)):
        return public__pb2.Value(string=str(val))
    if isinstance(val, int):
        return public__pb2.Value(int=val)
    if isinstance(val, float):
        return public__pb2.Value(double=val)
    else:
        err = f'Invalid value {value} of type {type(value)} for feature "{name}". Must be one of bool, str, float/int.'
        raise TypeError(err)


def _transform_label(value):
    if isinstance(value, public__pb2.Label):
        return value
    val = _convert_element(value)
    if isinstance(val, bool):
        return public__pb2.Label(binary=val)
    if isinstance(val, str):
        return public__pb2.Label(categorical=val)
    if isinstance(val, (int, float)):
        return public__pb2.Label(numeric=val)
    else:
        err = f'Invalid prediction/actual value {value} of type {type(value)}. Must be one of bool, str, float/int'
        raise TypeError(err)


def _convert_element(value):
    return getattr(value, "tolist", lambda: value)()
