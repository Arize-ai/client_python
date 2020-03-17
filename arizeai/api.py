import logging
import requests
from requests.exceptions import HTTPError
from aiohttp import ClientSession
from asyncio import get_event_loop

from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict

from . import protocol_pb2 as protocol__pb2

class API(object):
    """ 
    Synchronous API class to report model predictions and latent truths to Arize AI platform
    """
    def __init__(self, retry_attempts=3, timeout=200, api_key=None, account_id=None, uri='api.arize.com/v1/log'):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params account_id: (str) account id in Arize AI
        """
        assert api_key, 'API Key is required'
        assert account_id, 'Account ID is required'
        self._retry_attempts = retry_attempts
        self._uri = uri
        self._api_key = api_key
        self._account_id = account_id
        self._timeout = timeout
        self._LOGGER = logging.getLogger(__name__)

    def log(self, model_id: str, prediction_id: str, prediction_value=None, truth_value=None, labels={}):
        """ 
        :param model_id: (str) Unique identifier for a given model.
        :param prediction_id: (str) Unique indetifier for specific prediction. This is the key which latent truth events must tie back to.
        :param prediction_value: Mutually exclusive with truth_value. Output value for prediction (or latent truth). Can be bool, str, float, int.
        :param truth_value: See prediction_value.
        :param labels: (string, string) String dictionary containing prediction labels and/or metadata
        """
        assert model_id, 'model_id must be present when logging an event'
        assert prediction_id, 'prediction_id must be present when logging an event'
        record = self._build_record(model_id, prediction_id, prediction_value, truth_value, labels)
        json_record = MessageToDict(message=record, including_default_value_fields=False, preserving_proto_field_name=True)
        try:
            response = requests.post(
                self._uri,
                headers={'Authorization': self._api_key},
                timeout=self._timeout,
                json=json_record
            )
            response.raise_for_status()
        except HTTPError as http_err:
            self._process_exception(http_err)
        except Exception as err:
            self._LOGGER.error(f'Other error occurred: {err}') 

    def _build_record(self, model_id: str, prediction_id: str, prediction_value=None, truth_value=None, labels={}):
        record = None
        if prediction_value:
            record = self._build_prediction_record(
                model_id = model_id,
                prediction_id = prediction_id,
                prediction_value = prediction_value,
                labels = labels
            )
        elif truth_value:
            record = self._build_truth_record(
                model_id = model_id,
                prediction_id = prediction_id,
                truth_value = truth_value
            )
        if record is None:
            raise ValueError('prediction_value or truth_value must be present')
        return record

    def _build_prediction_record(self, model_id: str, prediction_id: str, prediction_value=None, labels={}):
        prediction =  protocol__pb2.Prediction(
            timestamp = self._get_time(),
            account_id = self._account_id,
            model_id = model_id,
            prediction_id = prediction_id,
            prediction_value = self._get_value(prediction_value),
            labels = labels
        )
        return protocol__pb2.Record(prediction=prediction)

    def _build_truth_record(self, model_id: str, prediction_id: str, truth_value=None):
        truth = protocol__pb2.Truth(
            timestamp = self._get_time(),
            account_id = self._account_id,
            model_id = model_id,
            prediction_id = prediction_id,
            truth_value = self._get_value(truth_value)
        )
        return protocol__pb2.Record(truth=truth)

    #TODO(gabe): Need to do something besides just log this
    def _process_exception(self, error):
        self._LOGGER.error(f"http error, while execute request:\n{error}") 

    ## TODO(gabe): Think of a better way to accomplish this, specifically I dont raising an exception, it's not very elegant
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


class AsyncAPI(API):
    """ 
    Asynchronous API class to report model predictions and latent truths to Arize AI platform
    """
    def __init__(self, *args, **kwargs):
        """
            :params api_key: (str) api key associated with your account with Arize AI
            :params account_id: (str) account id in Arize AI
        """
        super(AsyncAPI, self).__init__(*args,**kwargs)
        self.session = ClientSession(raise_for_status=True)
        self._loop = get_event_loop()

    async def _post(self, json_record):
        response = await self.session.post(
            self._uri,
            headers={'Authorization': self._api_key},
            timeout=self._timeout,
            json=json_record
        )
        async with response:
            assert response.status == 200


    def log(self, model_id: str, prediction_id: str, prediction_value=None, truth_value=None, labels={}):
        """ 
        :param model_id: (str) Unique identifier for a given model.
        :param prediction_id: (str) Unique indetifier for specific prediction. This is the key which latent truth events must tie back to.
        :param prediction_value: Mutually exclusive with truth_value. Output value for prediction (or latent truth). Can be bool, str, float, int.
        :param truth_value: See prediction_value.
        :param labels: (string, string) String dictionary containing prediction labels and/or metadata
        """
        assert model_id, 'model_id must be present when logging an event'
        assert prediction_id, 'prediction_id must be present when logging an event'
        record = self._build_record(model_id, prediction_id, prediction_value, truth_value, labels)
        json_record = MessageToDict(message=record, including_default_value_fields=False, preserving_proto_field_name=True)
        self._loop.create_task(self._post(json_record))