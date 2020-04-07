import logging
import requests
from requests.exceptions import HTTPError
from aiohttp import ClientSession, ClientResponseError, ClientTimeout
from asyncio import get_event_loop

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

from arize import protocol_pb2 as protocol__pb2

class Client(object):
    """ 
    Synchronous API Client to report model predictions and latent truths to Arize AI platform
    """
    def __init__(self, api_key: str, account_id: int, uri='https://api.arize.com/v1/log', retry_attempts=3, timeout=200):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params account_id: (int) account id in Arize AI
        """ 
        self._retry_attempts = retry_attempts
        self._uri = uri
        self._api_key = api_key
        self._account_id = account_id
        self._timeout = timeout
        self._LOGGER = logging.getLogger(__name__)

    def log(self, model_id: str, prediction_id: str, model_version=None, prediction_value=None, truth_value=None, labels=None):
        """ 
        :param model_id: (str) Unique identifier for a given model.
        :param prediction_id: (str) Unique indetifier for specific prediction. This is the key which latent truth events must tie back to.
        :param model_version: (str) Optional field used to group together a subset of predictions and truths for a given model_id.
        :param prediction_value: Mutually exclusive to truth_value. Output value for prediction (or latent truth). Can be bool, str, float, int.
        :param truth_value: Mutually exclusive to prediction_value. Latent truth value. Must be same type as original prediction_value (related by prediction_id).
        :param labels: (str, str) String dictionary containing prediction labels and/or metadata
        """
        try:
            assert model_id, 'model_id must be present when logging an event'
            assert prediction_id, 'prediction_id must be present when logging an event'
            record = self._build_record(model_id, prediction_id, model_version, prediction_value, truth_value, labels)
            json_record = MessageToDict(message=record, including_default_value_fields=False, preserving_proto_field_name=True)
            response = requests.post(
                self._uri,
                headers={'Authorization': self._api_key},
                timeout=self._timeout,
                json=json_record
            )
            response.raise_for_status()
        except Exception as err:
            self._handle_exception(err) 

    def _build_record(self, model_id: str, prediction_id: str, model_version=None, prediction_value=None, truth_value=None, labels=None):
        if prediction_value:
            record = self._build_prediction_record(
                model_version = model_version,
                prediction_value = prediction_value,
                labels = labels
            )
        elif truth_value:
            record = self._build_truth_record(
                truth_value = truth_value
            )
        else:
            raise ValueError('prediction_value or truth_value must be present')
        record.account_id = self._account_id
        record.model_id = model_id
        record.prediction_id = prediction_id
        return record

    def _build_prediction_record(self, model_version, prediction_value, labels):
        prediction =  protocol__pb2.Prediction(
            timestamp = self._get_time(),
            model_version = model_version,
            prediction_value = self._get_value(prediction_value),
            labels = labels
        )
        return protocol__pb2.Record(prediction=prediction)

    def _build_truth_record(self, truth_value):
        truth = protocol__pb2.Truth(
            timestamp = self._get_time(),
            truth_value = self._get_value(truth_value)
        )
        return protocol__pb2.Record(truth=truth)

    #TODO(gabe): Instrument metrics and expose to client
    def _handle_exception(self, err):
        type_ = type(err)
        if type_ is HTTPError:
            self._LOGGER.error(f"Http error, while executing request {err}")
        elif type_ is ClientResponseError:
            if err.status == 403:
                self._LOGGER.error(f'Invalid API key for account_id {self._account_id}: {err}')
        elif type_ is TypeError:
            self._LOGGER.error(f'Type error: {err}')
        elif type_ is AssertionError:
            self._LOGGER.error(f'Assertion error: {err}')
        elif type_ is ValueError:
            self._LOGGER.error(f'Value error: {err}')
        else:
            self._LOGGER.error(f'Unexpected error occured: {err}')

    @staticmethod
    def _get_value(value):
        if type(value) == bool:
            return protocol__pb2.Value(binary_value=value)
        if type(value) == str:
            return protocol__pb2.Value(categorical_value=value)
        if type(value) == float or int:
            return protocol__pb2.Value(numeric_value=value)
        else:
            raise TypeError('Value must be oneof bool, str, float/int')
            
    @staticmethod
    def _get_time():
        ts = Timestamp()
        ts.GetCurrentTime()
        return ts


class AsyncClient(Client):
    """ 
    Asynchronous API Client to report model predictions and latent truths to Arize AI platform
    """
    def __init__(self, *args, **kwargs):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params account_id: (str) account id in Arize AI
        """
        super(AsyncClient, self).__init__(*args,**kwargs)
        self._session = None
        self._loop = get_event_loop()
    
    async def _get_session(self):
        if self._session is None:
            timeout = ClientTimeout(total=60)
            self._session = ClientSession(timeout=timeout, raise_for_status=True)
        return self._session

    async def _post(self, json_record):
        session = await self._get_session()
        async with session.post(
            self._uri,
            headers={'Authorization': self._api_key},
            timeout=self._timeout,
            json=json_record
        ) as response:
            if response.status != 200:
                self._LOGGER.error(f'Response Error: {response}')

    def log(self, model_id: str, prediction_id: str, model_version=None, prediction_value=None, truth_value=None, labels=None):
        """ 
        :param model_id: (str) Unique identifier for a given model.
        :param prediction_id: (str) Unique indetifier for specific prediction. This is the key which latent truth events must tie back to.
        :param model_version: (str) Optional field used to group together a subset of predictions and truths for a given model_id.
        :param prediction_value: Mutually exclusive to truth_value. Output value for prediction (or latent truth). Can be bool, str, float, int.
        :param truth_value: Mutually exclusive to prediction_value. Latent truth value. Must be same type as original prediction_value (related by prediction_id).
        :param labels: (string, string) String dictionary containing prediction labels and/or metadata
        """
        try: 
            assert model_id, 'model_id must be present when logging an event'
            assert prediction_id, 'prediction_id must be present when logging an event'
            record = self._build_record(model_id, prediction_id, model_version, prediction_value, truth_value, labels)
            json_record = MessageToDict(message=record, including_default_value_fields=False, preserving_proto_field_name=True)
            self._loop.run_until_complete(self._post(json_record))
        except Exception as err:
            self._handle_exception(err)
