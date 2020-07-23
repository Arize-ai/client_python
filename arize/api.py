import math

from requests_futures.sessions import FuturesSession

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

from arize import public_pb2 as public__pb2
from arize.bounded_executor import BoundedExecutor
from arize.validation_helper import _validate_inputs
from arize.input_transformer import _normalize_inputs, _convert_time, _build_value_map, _transform_label


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
        self._session = FuturesSession(
            executor=BoundedExecutor(max_queue_bound, max_workers))

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
        :param features: (str, <value>) Optional dictionary or 2-D Pandas dataframe. Containing human readable and debuggable model features. For dict keys must be strings, values one of string, boolean, float, long for a single prediction. For bulk uploads, pass in a 2-D pandas dataframe where df.columns contain feature names. Must have same number of rows as prediction_ids, prediction_labels, actual_labels.
        :param features_name_overwrite: (list<str>) Optional list of strings that if present will overwrite features.columns values. Must contain the same number of elements as features.columns.
        :param model_id: (str) Unique identifier for a given model.
        :param model_version: (str) Optional field used to group together a subset of predictions and actuals for a given model_id.
        :param time_overwrite: (int) Optional field with unix epoch time in seconds to overwrite timestamp for prediction. If None, prediction uses current timestamp. For bulk uploads, pass in list<int> with same number of elements as prediction_labels. 
        :rtype : concurrent.futures.Future
        """
        records, uri = self._handle_log(prediction_ids, prediction_labels,
                                        actual_labels, features, model_id,
                                        model_version, features_name_overwrite,
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

    def _handle_log(self, prediction_ids, prediction_labels, actual_labels,
                    features, model_id, model_version, features_name_overwrite,
                    time_overwrite):
        if model_id is None:
            model_id = self._model_id
        if model_version is None:
            model_version = self._model_version
        is_bulk = _validate_inputs(prediction_ids, prediction_labels,
                                   actual_labels, features, model_id,
                                   model_version, features_name_overwrite,
                                   time_overwrite)
        uri = None
        records = []
        if is_bulk:
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
                                   prediction_id=prediction_ids,
                                   prediction_label=prediction_labels,
                                   features=features,
                                   actual_label=actual_labels,
                                   time_overwrite=time_overwrite))
        return records, uri

    def _build_bulk_record(self, model_id, model_version, prediction_ids,
                           prediction_labels, actual_labels, features,
                           features_name_overwrite, time_overwrite):

        records = []
        ids, pred_labels_df, actual_labels_df, features_df, time_df = _normalize_inputs(
            prediction_ids=prediction_ids,
            prediction_labels=prediction_labels,
            actual_labels=actual_labels,
            features=features,
            features_name_overwrite=features_name_overwrite,
            time_overwrite=time_overwrite)

        for i, v in enumerate(ids):
            if not isinstance(v[0], (str, bytes)):
                msg = f'prediction_id {v[0]} has type {type(v[0])}, but expected one of: str, bytes'
                raise ValueError(msg)
            f = None
            if features_df is not None:
                f = features_df[i]
            time = None
            if time_df is not None:
                time = time_df[0][i]
            if pred_labels_df is not None:
                records.append(
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
                      prediction_label, features, actual_label,
                      time_overwrite):
        organization_key = self._organization_key
        ts = None
        if time_overwrite is not None:
            ts = _convert_time(time_overwrite)
        else:
            ts = self._get_time()
        if actual_label is not None:
            record = self._build_actuals_record(
                ts=ts,
                organization_key=organization_key,
                model_id=model_id,
                prediction_id=prediction_id,
                label=_transform_label(actual_label))
        else:
            record = self._build_prediction_record(
                ts=ts,
                organization_key=organization_key,
                model_id=model_id,
                model_version=model_version,
                prediction_id=prediction_id,
                label=_transform_label(prediction_label),
                features=_build_value_map(features))
        return record

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
    def _get_time():
        ts = Timestamp()
        ts.GetCurrentTime()
        return ts

    @staticmethod
    def _num_chuncks(records):
        total_bytes = 0
        for r in records:
            total_bytes += r.ByteSize()
        num_of_bulk = math.ceil(total_bytes / 100000)
        recs_per_msg = math.ceil(len(records) / num_of_bulk)
        return recs_per_msg
