import arize.api as api
import arize.protocol_pb2 as protocol__pb2

expected = {'model':'model_v0',
    'api_key': 'API_KEY',
    'prediction_id':'prediction_0',
    'value_binary':True,
    'value_categorical':'arize',
    'value_numeric':20.20,
    'account_id':1234,
    'label' : {'label_0': 'value_0', 'label_1': 'value_1'}
    }

def test_api_initialization():
    try:
        api.API()
    except Exception as e:
        assert isinstance(e, TypeError)

    try:
        api.API(api_key='test')
    except Exception as client_id_exception:
        assert isinstance(client_id_exception, TypeError)
    
    try:
        api.API(account_id='test')
    except Exception as account_id_exception:
        assert isinstance(account_id_exception, TypeError)

def setup_client():
    return api.API(account_id=expected['account_id'], api_key=expected['api_key'])

def test_build_record_binary_prediction():

    client = setup_client()
    record = client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], prediction_value=expected['value_binary'], labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value

    assert record.prediction.account_id == 1234
    assert record.prediction.model_id == expected['model']
    assert record.prediction.labels == expected['label']
    assert record.prediction.prediction_value.binary_value == expected['value_binary']

def test_build_record_categorical_prediction():

    client = setup_client()
    record = client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], prediction_value=expected['value_categorical'], labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value

    assert record.prediction.account_id == 1234
    assert record.prediction.model_id == expected['model']
    assert record.prediction.labels == expected['label']
    assert record.prediction.prediction_value.categorical_value == expected['value_categorical']

def test_build_record_numeric_prediction():

    client = setup_client()
    record = client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], prediction_value=expected['value_numeric'], labels=expected['label'])
    
    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value
    
    assert record.prediction.account_id == 1234
    assert record.prediction.model_id == expected['model']
    assert record.prediction.labels == expected['label']
    assert record.prediction.prediction_value.numeric_value == expected['value_numeric']

def test_build_record_numeric_truth():

    client = setup_client()
    record = client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], truth_value=expected['value_numeric'], labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.truth) == protocol__pb2.Truth
    assert type(record.truth.truth_value) == protocol__pb2.Value

    assert record.truth.account_id == 1234
    assert record.truth.model_id == expected['model']
    assert record.truth.truth_value.numeric_value == expected['value_numeric']

def test_build_record_categorical_truth():

    client = setup_client()
    record = client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], truth_value=expected['value_categorical'], labels=expected['label'])
    
    assert type(record) == protocol__pb2.Record
    assert type(record.truth) == protocol__pb2.Truth
    assert type(record.truth.truth_value) == protocol__pb2.Value
    
    assert record.truth.account_id == 1234
    assert record.truth.model_id == expected['model']
    assert record.truth.truth_value.categorical_value == expected['value_categorical']

def test_build_record_binary_truth():

    client = setup_client()
    record = client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], truth_value=expected['value_binary'], labels=expected['label'])
    
    assert type(record) == protocol__pb2.Record
    assert type(record.truth) == protocol__pb2.Truth
    assert type(record.truth.truth_value) == protocol__pb2.Value
    
    assert record.truth.account_id == 1234
    assert record.truth.model_id == expected['model']
    assert record.truth.truth_value.binary_value == expected['value_binary']

def test_build_record_no_value():

    client = setup_client()
    try:
        client._build_record(model_id=expected['model'], prediction_id=expected['prediction_id'], labels=expected['label'])
    except ValueError as e:
        assert isinstance(e, ValueError)
