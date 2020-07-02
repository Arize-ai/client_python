import logging
import pandas as pd

import math

from requests_futures.sessions import FuturesSession

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

from arize import public_pb2 as public__pb2
from arize.bounded_executor import BoundedExecutor


class Client(object):
    """
    Arize API Client to report model predictions and actuals to Arize AI platform
    """
    def __init__(self,
                 api_key: str,
                 organization_key: str,
                 model_id=None,
                 model_version=None,
                 uri='https://api.arize.com/v1',
                 max_workers=8,
                 max_queue_bound=5000,
                 retry_attempts=3,
                 timeout=200):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params organization_key: (str) organization key in Arize AI
            :params model_id: (str) model id
            :params model_version: (str) model version
            :params max_workers: (int) number of max concurrent requests to Arize. Default: 20
            :max_queue_bound: (int) number of maximum concurrent future objects being generated for publishing to Arize. Default: 5000

        """
        self._retry_attempts = retry_attempts
        self._uri = uri + '/log'
        self._bulk_url = uri + '/bulk'
        self._api_key = api_key
        self._organization_key = organization_key
        self._model_id = model_id
        self._model_version = model_version
        self._timeout = timeout
        self._LOGGER = logging.getLogger(__name__)
        self._session = FuturesSession(
            executor=BoundedExecutor(max_queue_bound, max_workers))

    # TODO: Drop time_overwrite prior 0.1.0 release
    def log(self,
            prediction_ids,
            prediction_labels=None,
            actual_labels=None,
            features=None,
            features_name_overwrite=None,
            model_id=None,
            model_version=None,
            time_overwrite=None):
        """ Logs an event with Arize via a POST request. Returns :class:`Future` object.
        :param prediction_ids: (str) Unique string indetifier for specific prediction or actual label. These values are needed to match latent actual labels to their original prediction labels. For bulk uploads, pass in a 1-D Pandas Series where values are ids corresponding to feature values of the same index.
        :param prediction_labels: The predicted values for a given model input. Individual labels can be joined against actual_labels by the corresponding prediction id. For individual events, input values can be bool, str, float, int. For bulk uploads, the client accepts a 1-D pandas data frame where values are associates to the label values in the same index. Must be the same shape as actual_labels.
        :param actual_labels: The actual expected values for a given model input. Individual labels can be joined against prediction_labels by the corresponding prediction id. For individual events, input values can be bool, str, float, int. For bulk uploads, the client accepts a 1-D pandas data frame where values are associates to the label values in the same index. Must be the same shape as prediction_labels.
        :param features: (str, <value>) Dictionary or 2-D Pandas dataframe. Containing human readable and debuggable model features. For dict keys must be strings, values one of string, boolean, float, long for a single prediction. For bulk uploads, pass in a 2-D pandas dataframe where df.columns contain feature names. Must have same number of rows as prediction_ids, prediction_labels, actual_labels.
        :param features_name_overwrite: (list<str>) Optional list of strings that if present will overwrite features.columns values. Must contain the same number of elements as features.columns.
        :param model_id: (str) Unique identifier for a given model.
        :param model_version: (str) Optional field used to group together a subset of predictions and actuals for a given model_id.
        :rtype : concurrent.futures.Future
        """
        try:
            records, uri = self._handle_log(prediction_ids, prediction_labels,
                                            actual_labels, features, model_id,
                                            model_version,
                                            features_name_overwrite,
                                            time_overwrite)
            responses = []
            for record in records:
                payload = MessageToDict(message=record,
                                        preserving_proto_field_name=True)
                response = self._session.post(
                    uri,
                    headers={'Authorization': self._api_key},
                    timeout=self._timeout,
                    json=payload)
                responses.append(response)
            return responses
        except Exception as err:
            self._handle_exception(err)

    def _handle_log(self, prediction_ids, prediction_labels, actual_labels,
                    features, model_id, model_version, features_name_overwrite,
                    time_overwrite):
        uri = None
        records = []
        if model_id is None:
            model_id = self._model_id
        if model_version is None:
            model_version = self._model_version
        if prediction_labels is None and actual_labels is None:
            raise ValueError(
                'either prediction_labels or actual_labels must be passed in, both are None'
            )
        if isinstance(prediction_ids, pd.DataFrame):
            if prediction_labels is not None:
                self._validate_bulk_prediction_inputs(prediction_ids,
                                                      prediction_labels,
                                                      features,
                                                      features_name_overwrite,
                                                      time_overwrite)
            if actual_labels is not None:
                self._validate_bulk_actuals_inputs(prediction_ids,
                                                   actual_labels)
            uri = self._bulk_url
            records = self._build_bulk_record(
                model_id=model_id,
                model_version=model_version,
                prediction_ids=prediction_ids,
                prediction_labels=prediction_labels,
                actual_labels=actual_labels,
                features=features,
                features_name_overwrite=features_name_overwrite,
                time_overwrite=time_overwrite)
        else:
            uri = self._uri
            records.append(
                self._build_record(model_id=model_id,
                                   model_version=model_version,
                                   prediction_id=str(prediction_ids),
                                   prediction_label=prediction_labels,
                                   features=features,
                                   actual_label=actual_labels))
        return records, uri

    def _build_bulk_record(self, model_id, model_version, prediction_ids,
                           prediction_labels, actual_labels, features,
                           features_name_overwrite, time_overwrite):
        records = []
        ids = prediction_ids.to_numpy()
        pred_labels_df = None
        actual_labels_df = None
        features_df = None
        time_df = None
        if prediction_labels is not None:
            pred_labels_df = prediction_labels.applymap(
                lambda x: self._get_label(x)).to_numpy()
        if actual_labels is not None:
            actual_labels_df = actual_labels.applymap(
                lambda x: self._get_label(x)).to_numpy()
        if features is not None:
            if features_name_overwrite is not None:
                features.columns = features_name_overwrite
            features_df = self._build_value_map(features).to_dict('records')

        #TODO: Strip this before 0.1.0
        if time_overwrite is not None:
            time_df = time_overwrite.applymap(lambda x: self._convert_time(x))

        for i, v in enumerate(ids):
            f = None
            if features_df is not None:
                f = features_df[i]

            # TODO: strip time_overwrite before release
            time = None
            if time_df is not None:
                time = time_df[0][i]
            if pred_labels_df is not None:
                records.append(
                    # TODO: strip time_overwrite before release
                    self._build_prediction_record(ts=time,
                                                  organization_key=None,
                                                  model_id=None,
                                                  model_version=None,
                                                  prediction_id=v[0],
                                                  label=pred_labels_df[i][0],
                                                  features=f))
            if actual_labels_df is not None:
                records.append(
                    self._build_actuals_record(ts=None,
                                               organization_key=None,
                                               model_id=None,
                                               prediction_id=v[0],
                                               label=actual_labels_df[i][0]))

        recs_per_msg = self._num_chuncks(records)
        recs = [
            records[i:i + recs_per_msg]
            for i in range(0, len(records), recs_per_msg)
        ]
        results = [
            public__pb2.BulkRecord(records=r,
                                   organization_key=self._organization_key,
                                   model_id=model_id,
                                   model_version=model_version,
                                   timestamp=self._get_time()) for r in recs
        ]
        return results

    def _build_record(self, model_id, model_version, prediction_id,
                      prediction_label, features, actual_label):
        organization_key = self._organization_key
        ts = self._get_time()
        if actual_label is not None:
            record = self._build_actuals_record(
                ts=ts,
                organization_key=organization_key,
                model_id=model_id,
                prediction_id=prediction_id,
                label=self._get_label(actual_label))
        else:
            record = self._build_prediction_record(
                ts=ts,
                organization_key=organization_key,
                model_id=model_id,
                model_version=model_version,
                prediction_id=prediction_id,
                label=self._get_label(prediction_label),
                features=self._build_value_map(features))
        return record

    def _build_value_map(self, vals):
        formatted = None
        if isinstance(vals, dict):
            formatted = {k: self._get_value(v, k) for (k, v) in vals.items()}
        elif isinstance(vals, pd.DataFrame):
            formatted = vals.apply(
                lambda y: y.apply(lambda x: self._get_value(x, y.name)))
        return formatted

    # TODO(gabe): Instrument metrics and expose to client
    def _handle_exception(self, err):
        type_ = type(err)
        if type_ is TypeError:
            self._LOGGER.error(f'Type error: {err}')
        elif type_ is AssertionError:
            self._LOGGER.error(f'Assertion error: {err}')
        elif type_ is ValueError:
            self._LOGGER.error(f'Value error: {err}')
        else:
            self._LOGGER.error(f'Unexpected error occured: {err}')

    @staticmethod
    def _build_prediction_record(ts, organization_key, model_id, model_version,
                                 prediction_id, label, features):
        p = public__pb2.Prediction(label=label, features=features)
        if isinstance(ts, Timestamp):
            p.timestamp.MergeFrom(ts)
        if model_version is not None:
            p.model_version = model_version
        rec = public__pb2.Record(prediction_id=prediction_id, prediction=p)
        if organization_key is not None:
            rec.organization_key = organization_key
        if model_id is not None:
            rec.model_id = model_id
        return rec

    @staticmethod
    def _build_actuals_record(ts, organization_key, model_id, prediction_id,
                              label):
        actual = public__pb2.Actual(label=label)
        if isinstance(ts, Timestamp):
            actual.timestamp.MergeFrom(ts)
        rec = public__pb2.Record(prediction_id=prediction_id, actual=actual)
        if organization_key is not None:
            rec.organization_key = organization_key
        if model_id is not None:
            rec.model_id = model_id
        return rec

    @staticmethod
    def _get_label(value):
        if isinstance(value, public__pb2.Label):
            return value
        val = Client._convert_element(value)
        if isinstance(val, bool):
            return public__pb2.Label(binary=val)
        if isinstance(val, str):
            return public__pb2.Label(categorical=val)
        if isinstance(val, (int, float)):
            return public__pb2.Label(numeric=val)
        else:
            err = f'Invalid prediction/actual value {value} of type {type(value)}. Must be one of bool, str, float/int'
            raise TypeError(err)

    @staticmethod
    def _get_value(value, name):
        if isinstance(value, public__pb2.Value):
            return value
        val = Client._convert_element(value)
        if isinstance(val, (str, bool)):
            return public__pb2.Value(string=str(val))
        if isinstance(val, int):
            return public__pb2.Value(int=val)
        if isinstance(val, float):
            return public__pb2.Value(double=val)
        else:
            err = f'Invalid value {value} of type {type(value)} for feature "{name}". Must be one of bool, str, float/int.'
            raise TypeError(err)

    @staticmethod
    def _get_time():
        ts = Timestamp()
        ts.GetCurrentTime()
        return ts

    @staticmethod
    def _convert_element(value):
        return getattr(value, "tolist", lambda: value)()

    @staticmethod
    def _validate_bulk_prediction_inputs(prediction_ids, prediction_labels,
                                         features, features_name_overwrite,
                                         timestamp_overwrite):
        #TODO: strip timestamp override prior to 0.1.0
        if timestamp_overwrite is not None:
            if timestamp_overwrite.shape[0] != prediction_ids.shape[0]:
                msg = f'timestamp_overwrite has {timestamp_overwrite.shape[0]} but must have same number of elements as prediction_ids {prediction_ids.shape[0]}'
                raise ValueError(msg)
        if prediction_ids is None:
            raise ValueError('at least one prediction id is required')
        if prediction_labels is None:
            raise ValueError('at least one prediction label is required')
        if prediction_labels.shape[0] != prediction_ids.shape[0]:
            msg = f'prediction_labels shaped {prediction_labels.shape[0]} must have the same number of rows as predictions_ids shaped {prediction_ids.shape[0]}.'
            raise ValueError(msg)
        if features is not None and features.shape[0] != prediction_ids.shape[
                0]:
            msg = f'features shaped {features.shape[0]} must have the same number of rows as predictions_ids shaped {prediction_ids.shape[0]}.'
            raise ValueError(msg)
        if features_name_overwrite is not None and len(
                features.columns) != len(features_name_overwrite):
            msg = f'features_name_overwrite len:{len(features_name_overwrite)} must have the same number of elements as features has columns ({len(features.columns)} columns).'
            raise ValueError(msg)
        if features is not None and isinstance(
                features.columns, pd.core.indexes.range.RangeIndex
        ) and features_name_overwrite is None:
            msg = f'fatures.columns is of type RangeIndex, therefore, features_name_overwrite must be present to overwrite columns index with human readable feature names.'
            raise ValueError(msg)

    @staticmethod
    def _validate_bulk_actuals_inputs(prediction_ids, actual_labels):
        if prediction_ids is None:
            raise ValueError('at least one prediction id is required')
        if actual_labels is None:
            raise ValueError('at least one actual label is required')
        if actual_labels.shape[0] != prediction_ids.shape[0]:
            msg = f'actual_labels shaped {actual_labels.shape[0]} must have the same number of rows as predictions_ids shaped {prediction_ids.shape[0]}.'
            raise ValueError(msg)

    @staticmethod
    def _num_chuncks(records):
        total_bytes = 0
        for r in records:
            total_bytes += r.ByteSize()
        num_of_bulk = math.ceil(total_bytes / 100000)
        recs_per_msg = math.ceil(len(records) / num_of_bulk)
        return recs_per_msg

    @staticmethod
    def _convert_time(time):
        ts = None
        if time is not None:
            ts = Timestamp()
            ts.FromSeconds(time)
        return ts
