import logging
import pandas as pd

import math

from requests_futures.sessions import FuturesSession

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

from arize import protocol_pb2 as protocol__pb2
from arize.bounded_executor import BoundedExecutor


class Client(object):
    """
    Arize API Client to report model predictions and latent truths to Arize AI platform
    """
    def __init__(self,
                 api_key: str,
                 account_id: int,
                 model_id=None,
                 model_version=None,
                 uri='https://api.arize.com/v1',
                 max_workers=8,
                 max_queue_bound=5000,
                 retry_attempts=3,
                 timeout=200):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params account_id: (int) account id in Arize AI
            :params model_id: (str) model id
            :params model_version: (str) model version
            :params max_workers: (int) number of max concurrent requests to Arize. Default: 20
            :max_queue_bound: (int) number of maximum concurrent future objects being generated for publishing to Arize. Default: 5000

        """
        self._retry_attempts = retry_attempts
        self._uri = uri + '/log'
        self._bulk_url = uri + '/bulk'
        self._api_key = api_key
        self._account_id = account_id
        self._model_id = model_id
        self._model_version = model_version
        self._timeout = timeout
        self._LOGGER = logging.getLogger(__name__)
        self._session = FuturesSession(
            executor=BoundedExecutor(max_queue_bound, max_workers))

    def log(self,
            prediction_ids,
            values,
            labels,
            is_latent_truth,
            model_id=None,
            model_version=None):
        """ Logs an event with Arize via a POST request. Returns :class:`Future` object.
        :param prediction_ids: (str) Unique indetifier for specific prediction. This is the key which latent truth events must tie back to. For bulk uploads, pass in a 1-D pandas df where values are ids corresponding to label values in the same index.
        :param values: The prediction or latent truth value which can be joined via prediction id. Can be bool, str, float, int. For bulk uploads, pass in a 1-D pandas data frame where values correspond to the label values in the same index.
        :param labels: (str, <value>) Dictionary or 2-D Pandas dataframe. containing prediction labels and/or metadata. For dict keys must be strings, values oneof string, boolean, float, long for a single prediction. For bulk uploads, pass in a 2-D pandas dataframe where df.columns contain label names.
        :param is_latent_truth: (bool) Flag identifying if values being logged are latent truths
        :param model_id: (str) Unique identifier for a given model.
        :param model_version: (str) Optional field used to group together a subset of predictions and truths for a given model_id.
        :rtype : concurrent.futures.Future
        """
        try:
            if values is None:
                raise ValueError('a value is required')
            uri = None
            records = []
            if model_id is None:
                model_id = self._model_id
            if model_version is None:
                model_version = self._model_version
            if isinstance(labels, pd.DataFrame):
                uri = self._bulk_url
                records = self._build_bulk_record(
                    model_id=model_id,
                    model_version=model_version,
                    prediction_ids=prediction_ids,
                    values=values,
                    labels=labels,
                    latent_truth=is_latent_truth)
            elif isinstance(labels, dict):
                uri = self._uri
                records.append(
                    self._build_record(model_id=model_id,
                                       model_version=model_version,
                                       prediction_id=prediction_ids,
                                       values=values,
                                       labels=labels,
                                       latent_truth=is_latent_truth))
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

    def _build_bulk_record(self,
                           model_id: str,
                           latent_truth: bool,
                           prediction_ids=None,
                           model_version=None,
                           values=None,
                           labels=None):
        records = []
        ids = prediction_ids.to_numpy()
        value_df = values.applymap(lambda x: self._get_value(x)).to_numpy()
        record = None
        if latent_truth:
            for i, v in enumerate(value_df):
                truth = protocol__pb2.Truth(truth_value=v[0])
                record = protocol__pb2.Record(prediction_id=ids[i][0],
                                              truth=truth)
                records.append(record)
        else:
            labels_df = self._build_label_map(labels).to_dict('records')
            for i, v in enumerate(value_df):
                pred = protocol__pb2.Prediction(
                    prediction_value=v[0],
                    labels=labels_df[i],
                )
                record = protocol__pb2.Record(prediction_id=ids[i][0],
                                              prediction=pred)
                records.append(record)
        total_bytes = 0
        for r in records:
            total_bytes += r.ByteSize()
        num_of_bulk = math.ceil(total_bytes / 100000)
        recs_per_msg = math.ceil(len(records) / num_of_bulk)
        recs = [
            records[i:i + recs_per_msg]
            for i in range(0, len(records), recs_per_msg)
        ]
        results = [
            protocol__pb2.BulkRecord(records=r,
                                     account_id=self._account_id,
                                     model_id=model_id,
                                     model_version=model_version,
                                     timestamp=self._get_time()) for r in recs
        ]
        return results

    def _build_record(self,
                      model_id: str,
                      prediction_id: str,
                      latent_truth: bool,
                      model_version=None,
                      values=None,
                      labels=None):

        account_id = self._account_id
        val = self._get_value(values)
        ts = self._get_time()
        if model_version is None:
            model_version = self._model_version
        if latent_truth:
            record = self._build_truth_record(ts=ts,
                                              account_id=account_id,
                                              model_id=model_id,
                                              prediction_id=prediction_id,
                                              value=val)
        else:
            record = self._build_prediction_record(
                ts=ts,
                model_version=model_version,
                value=val,
                labels=self._build_label_map(labels),
                account_id=account_id,
                model_id=model_id,
                prediction_id=prediction_id)
        return record

    def _build_label_map(self, labels):
        formatted_labels = None
        if isinstance(labels, dict):
            formatted_labels = {
                k: self._get_label_value(v, k)
                for (k, v) in labels.items()
            }
        elif isinstance(labels, pd.DataFrame):
            formatted_labels = labels.apply(
                lambda y: y.apply(lambda x: self._get_label_value(x, y.name)))
        return formatted_labels

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
    def _build_prediction_record(ts, model_version, value, labels, account_id,
                                 model_id, prediction_id):
        prediction = protocol__pb2.Prediction(timestamp=ts,
                                              model_version=model_version,
                                              prediction_value=value,
                                              labels=labels)
        return protocol__pb2.Record(account_id=account_id,
                                    model_id=model_id,
                                    prediction_id=prediction_id,
                                    prediction=prediction)

    @staticmethod
    def _build_truth_record(ts, account_id, model_id, prediction_id, value):
        truth = protocol__pb2.Truth(timestamp=ts, truth_value=value)
        return protocol__pb2.Record(account_id=account_id,
                                    model_id=model_id,
                                    prediction_id=prediction_id,
                                    truth=truth)

    @staticmethod
    def _get_value(value):
        val = Client._convert_value(value)
        if isinstance(val, bool):
            return protocol__pb2.Value(binary_value=val)
        if isinstance(val, str):
            return protocol__pb2.Value(categorical_value=val)
        if isinstance(val, (int, float)):
            return protocol__pb2.Value(numeric_value=val)
        else:
            err = f'Invalid prediction value {value} of type {type(value)}. Must be one of bool, str, float/int'
            raise TypeError(err)

    @staticmethod
    def _get_label_value(value, label_name):
        val = Client._convert_value(value)
        if isinstance(val, (str, bool)):
            return protocol__pb2.Label(string_label=str(val))
        if isinstance(val, int):
            return protocol__pb2.Label(int_label=val)
        if isinstance(val, float):
            return protocol__pb2.Label(double_label=val)
        else:
            err = f'Invalid value {value} of type {type(value)} for label "{label_name}". Must be one of bool, str, float/int.'
            raise TypeError(err)

    @staticmethod
    def _get_time():
        ts = Timestamp()
        ts.GetCurrentTime()
        return ts

    @staticmethod
    def _convert_value(value):
        return getattr(value, "tolist", lambda: value)()
