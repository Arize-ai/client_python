import arize.api as api
import arize.protocol_pb2 as protocol__pb2

NUM_VAL = 20.20
STR_VAL = 'arize'
BOOL_VAL = True
INT_VAL = 0

expected = {
    'model': 'model_v0',
    'model_version': 'v1.2.3.4',
    'api_key': 'API_KEY',
    'prediction_id': 'prediction_0',
    'value_binary': BOOL_VAL,
    'value_categorical': STR_VAL,
    'value_numeric': NUM_VAL,
    'account_id': 1234,
    'label': {
        'label_bool': BOOL_VAL,
        'label_str': STR_VAL,
        'label_float': NUM_VAL,
        'label_int': INT_VAL
    }
}


def test_api_initialization():
    try:
        api.Client()
    except Exception as e:
        assert isinstance(e, TypeError)

    try:
        api.Client(api_key='test')
    except Exception as client_id_exception:
        assert isinstance(client_id_exception, TypeError)

    try:
        api.Client(account_id='test')
    except Exception as account_id_exception:
        assert isinstance(account_id_exception, TypeError)


def setup_client():
    return api.Client(account_id=expected['account_id'],
                      api_key=expected['api_key'])


def test_build_record_labels():

    client = setup_client()
    record = client._build_record(model_id=expected['model'],
                                  model_version=expected['model_version'],
                                  prediction_id=expected['prediction_id'],
                                  prediction_value=expected['value_binary'],
                                  labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value
    assert type(record.prediction.labels['label_bool']) == protocol__pb2.Label
    assert type(record.prediction.labels['label_str']) == protocol__pb2.Label
    assert type(record.prediction.labels['label_float']) == protocol__pb2.Label
    assert type(record.prediction.labels['label_int']) == protocol__pb2.Label

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert record.prediction.prediction_value.binary_value == expected[
        'value_binary']

    assert record.prediction.labels['label_bool'].WhichOneof(
        'label_value') == 'string_label'
    assert record.prediction.labels['label_str'].WhichOneof(
        'label_value') == 'string_label'
    assert record.prediction.labels['label_float'].WhichOneof(
        'label_value') == 'double_label'
    assert record.prediction.labels['label_int'].WhichOneof(
        'label_value') == 'int_label'


def test_build_record_binary_prediction():

    client = setup_client()
    record = client._build_record(model_id=expected['model'],
                                  model_version=expected['model_version'],
                                  prediction_id=expected['prediction_id'],
                                  prediction_value=expected['value_binary'],
                                  labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert record.prediction.labels is not None
    assert record.prediction.prediction_value.binary_value == expected[
        'value_binary']


def test_build_record_categorical_prediction():

    client = setup_client()
    record = client._build_record(
        model_id=expected['model'],
        model_version=expected['model_version'],
        prediction_id=expected['prediction_id'],
        prediction_value=expected['value_categorical'],
        labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert record.prediction.labels is not None
    assert record.prediction.prediction_value.categorical_value == expected[
        'value_categorical']


def test_build_record_numeric_prediction():

    client = setup_client()
    record = client._build_record(model_id=expected['model'],
                                  prediction_id=expected['prediction_id'],
                                  model_version=expected['model_version'],
                                  prediction_value=expected['value_numeric'],
                                  labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.prediction) == protocol__pb2.Prediction
    assert type(record.prediction.prediction_value) == protocol__pb2.Value

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.prediction.model_version == expected['model_version']
    assert record.prediction.labels is not None
    assert record.prediction.prediction_value.numeric_value == expected[
        'value_numeric']


def test_build_record_numeric_truth():

    client = setup_client()
    record = client._build_record(model_id=expected['model'],
                                  prediction_id=expected['prediction_id'],
                                  truth_value=expected['value_numeric'],
                                  labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.truth) == protocol__pb2.Truth
    assert type(record.truth.truth_value) == protocol__pb2.Value

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.truth.truth_value.numeric_value == expected['value_numeric']


def test_build_record_categorical_truth():

    client = setup_client()
    record = client._build_record(model_id=expected['model'],
                                  prediction_id=expected['prediction_id'],
                                  truth_value=expected['value_categorical'],
                                  labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.truth) == protocol__pb2.Truth
    assert type(record.truth.truth_value) == protocol__pb2.Value

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.truth.truth_value.categorical_value == expected[
        'value_categorical']


def test_build_record_binary_truth():

    client = setup_client()
    record = client._build_record(model_id=expected['model'],
                                  prediction_id=expected['prediction_id'],
                                  truth_value=expected['value_binary'],
                                  labels=expected['label'])

    assert type(record) == protocol__pb2.Record
    assert type(record.truth) == protocol__pb2.Truth
    assert type(record.truth.truth_value) == protocol__pb2.Value

    assert record.account_id == expected['account_id']
    assert record.model_id == expected['model']
    assert record.prediction_id == expected['prediction_id']
    assert record.truth.truth_value.binary_value == expected['value_binary']


def test_build_record_no_value():

    client = setup_client()
    try:
        client._build_record(model_id=expected['model'],
                             prediction_id=expected['prediction_id'],
                             labels=expected['label'])
    except ValueError as e:
        assert isinstance(e, ValueError)
