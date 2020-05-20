import logging
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
                 uri='https://api.arize.com/v1/log',
                 max_workers=40,
                 max_queue_bound=5000,
                 retry_attempts=3,
                 timeout=200):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params account_id: (int) account id in Arize AI
            :params max_workers: (int) number of max concurrent requests to Arize. Default: 40
            :max_queue_bound: (int) number of maximum concurrent future objects being generated for publishing to Arize. Default: 5000

        """
        self._retry_attempts = retry_attempts
        self._uri = uri
        self._api_key = api_key
        self._account_id = account_id
        self._timeout = timeout
        self._LOGGER = logging.getLogger(__name__)
        self._session = FuturesSession(
            executor=BoundedExecutor(max_queue_bound, max_workers))

    def log(self,
            model_id: str,
            prediction_id: str,
            model_version=None,
            prediction_value=None,
            truth_value=None,
            labels=None):
        """ Logs an event with Arize via a POST request. Returns :class:`Future` object.
        :param model_id: (str) Unique identifier for a given model.
        :param prediction_id: (str) Unique indetifier for specific prediction. This is the key which latent truth events must tie back to.
        :param model_version: (str) Optional field used to group together a subset of predictions and truths for a given model_id.
        :param prediction_value: Mutually exclusive to truth_value. Output value for prediction (or latent truth). Can be bool, str, float, int.
        :param truth_value: Mutually exclusive to prediction_value. Latent truth value. Must be same type as original prediction_value (related by prediction_id).
        :param labels: (str, <value>) Dictionary containing prediction labels and/or metadata. Keys must be strings, values oneof string, boolean, float, long.
        :rtype : concurrent.futures.Future
        """
        try:
            assert model_id, 'model_id must be present when logging an event'
            assert prediction_id, 'prediction_id must be present when logging an event'
            record = self._build_record(model_id, prediction_id, model_version,
                                        prediction_value, truth_value, labels)
            json_record = MessageToDict(message=record,
                                        including_default_value_fields=False,
                                        preserving_proto_field_name=True)
            response = self._session.post(
                self._uri,
                headers={'Authorization': self._api_key},
                timeout=self._timeout,
                json=json_record)
            return response
        except Exception as err:
            self._handle_exception(err)

    def _build_record(self,
                      model_id: str,
                      prediction_id: str,
                      model_version=None,
                      prediction_value=None,
                      truth_value=None,
                      labels=None):
        if prediction_value is not None:
            record = self._build_prediction_record(
                model_version=model_version,
                prediction_value=prediction_value,
                labels=labels)
        elif truth_value is not None:
            record = self._build_truth_record(truth_value=truth_value)
        else:
            raise ValueError(
                'prediction_value or truth_value must be present. Values passed in as NoneType'
            )
        record.account_id = self._account_id
        record.model_id = model_id
        record.prediction_id = prediction_id
        return record

    def _build_prediction_record(self, model_version, prediction_value,
                                 labels):
        prediction = protocol__pb2.Prediction(
            timestamp=self._get_time(),
            model_version=model_version,
            prediction_value=self._get_value(prediction_value, True),
            labels=self._build_label_map(labels))
        return protocol__pb2.Record(prediction=prediction)

    def _build_label_map(self, labels):
        formatted_labels = {}
        for k, v in labels.items():
            formatted_labels[k] = self._get_label_value(v, k)
        return formatted_labels

    def _build_truth_record(self, truth_value):
        truth = protocol__pb2.Truth(timestamp=self._get_time(),
                                    truth_value=self._get_value(
                                        truth_value, False))
        return protocol__pb2.Record(truth=truth)

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
    def _get_value(value, isPred):
        val, value_type = Client._convert_value(value)
        if value_type == bool:
            return protocol__pb2.Value(binary_value=val)
        if value_type == str:
            return protocol__pb2.Value(categorical_value=val)
        if value_type == float or int:
            return protocol__pb2.Value(numeric_value=val)
        else:
            err = None
            if isPred:
                err = f'Invalid prediction_value {value} of type {value_type}. Must be one of bool, str, float/int'
            else:
                err = f'Invalid truth_value {value} of type {value_type}. Must be one of bool, str, float/int'
            raise TypeError(err)

    @staticmethod
    def _get_label_value(value, label_name):
        label_value, label_type = Client._convert_value(value)
        if label_type == str:
            return protocol__pb2.Label(string_label=label_value)
        if label_type == int:
            return protocol__pb2.Label(int_label=label_value)
        if label_type == float:
            return protocol__pb2.Label(double_label=label_value)
        if label_type == bool:
            return protocol__pb2.Label(string_label=str(label_value).lower())
        else:
            err = f'Invalid label_value {label_value} of type {label_type} for label "{label_name}". Must be one of bool, str, float/int.'
            raise TypeError(err)

    @staticmethod
    def _get_time():
        ts = Timestamp()
        ts.GetCurrentTime()
        return ts

    @staticmethod
    def _convert_value(value):
        converted_value = getattr(value, "tolist", lambda: value)()
        type_ = type(converted_value)
        return converted_value, type_
