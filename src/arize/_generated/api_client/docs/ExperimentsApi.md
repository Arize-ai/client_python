# arize._generated.api_client.ExperimentsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**experiments_create**](ExperimentsApi.md#experiments_create) | **POST** /v2/experiments | Create an experiment
[**experiments_delete**](ExperimentsApi.md#experiments_delete) | **DELETE** /v2/experiments/{experiment_id} | Delete an experiment
[**experiments_get**](ExperimentsApi.md#experiments_get) | **GET** /v2/experiments/{experiment_id} | Get an experiment
[**experiments_list**](ExperimentsApi.md#experiments_list) | **GET** /v2/experiments | List experiments
[**experiments_runs_list**](ExperimentsApi.md#experiments_runs_list) | **GET** /v2/experiments/{experiment_id}/runs | List experiment runs


# **experiments_create**
> Experiment experiments_create(experiments_create_request)

Create an experiment

Create a new experiment. Empty experiments are not allowed.

Experiments are composed of "runs". Each experiment run (JSON object)
must include an `example_id` field that corresponds to an example in
the dataset, and a `output` field that contains the task's output for
the example (the input).

Payload Requirements
- The `name` must be unique within the target dataset
- Provide at least one run in `experiment_runs`.
- Each run must include:
  - `example_id` -- the ID of an existing example in the dataset/version
  - `output` -- model/task output for that example
  - You may include any additional fields per run that can be used for
  analysis or filtering. For exampple: `model`, `latency_ms`,
  `temperature`, `prompt`, `tool_calls`, etc.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.experiment import Experiment
from arize._generated.api_client.models.experiments_create_request import ExperimentsCreateRequest
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.ExperimentsApi(api_client)
    experiments_create_request = {"name":"My Experiment Name","dataset_id":"dataset_12345","experiment_runs":[{"example_id":"example_001","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":118,"prompt":"Answer the math question briefly."},{"example_id":"example_002","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":132},{"example_id":"example_003","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":125}]} # ExperimentsCreateRequest | Body containing experiment creation parameters

    try:
        # Create an experiment
        api_response = api_instance.experiments_create(experiments_create_request)
        print("The response of ExperimentsApi->experiments_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->experiments_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiments_create_request** | [**ExperimentsCreateRequest**](ExperimentsCreateRequest.md)| Body containing experiment creation parameters | 

### Return type

[**Experiment**](Experiment.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | An experiment object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**409** | Resource conflict |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **experiments_delete**
> experiments_delete(experiment_id)

Delete an experiment

Delete an experiment by its ID. This operation is irreversible.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.ExperimentsApi(api_client)
    experiment_id = 'experiment_id_example' # str | The unique identifier of the experiment

    try:
        # Delete an experiment
        api_instance.experiments_delete(experiment_id)
    except Exception as e:
        print("Exception when calling ExperimentsApi->experiments_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique identifier of the experiment | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**204** | Experiment successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **experiments_get**
> Experiment experiments_get(experiment_id)

Get an experiment

Get an experiment object by its ID.

The response does not include the experiment's runs. To get the runs of
a specific experiment, use the List Experiment Runs endpoint.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.experiment import Experiment
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.ExperimentsApi(api_client)
    experiment_id = 'experiment_id_example' # str | The unique identifier of the experiment

    try:
        # Get an experiment
        api_response = api_instance.experiments_get(experiment_id)
        print("The response of ExperimentsApi->experiments_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->experiments_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique identifier of the experiment | 

### Return type

[**Experiment**](Experiment.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An experiment object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **experiments_list**
> ExperimentsList200Response experiments_list(dataset_id=dataset_id, limit=limit, cursor=cursor)

List experiments

List all experiments a user has access to.

To filter experiments by the dataset they were run on, provide the
`dataset_id` query parameter.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.experiments_list200_response import ExperimentsList200Response
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.ExperimentsApi(api_client)
    dataset_id = 'dataset_id_example' # str | Filter experiments to a particular dataset ID (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List experiments
        api_response = api_instance.experiments_list(dataset_id=dataset_id, limit=limit, cursor=cursor)
        print("The response of ExperimentsApi->experiments_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->experiments_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| Filter experiments to a particular dataset ID | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ExperimentsList200Response**](ExperimentsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of experiment objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **experiments_runs_list**
> ExperimentsRunsList200Response experiments_runs_list(experiment_id, limit=limit)

List experiment runs

List runs for a given experiment.

The runs are sorted by insertion order.

**Pagination**:
- Response includes `pagination` for forward compatibility.
- **Currently not implemented**: `pagination.next_cursor` is omitted
- When pagination is enabled in the future, the behavior will match
other list endpoints (cursor-based, opaque tokens).

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.experiments_runs_list200_response import ExperimentsRunsList200Response
from arize._generated.api_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to https://api.arize.com
# See configuration.py for a list of all supported configuration parameters.
configuration = arize._generated.api_client.Configuration(
    host = "https://api.arize.com"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure Bearer authorization (<api-key>): bearerAuth
configuration = arize._generated.api_client.Configuration(
    access_token = os.environ["BEARER_TOKEN"]
)

# Enter a context with an instance of the API client
with arize._generated.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = arize._generated.api_client.ExperimentsApi(api_client)
    experiment_id = 'experiment_id_example' # str | The unique identifier of the experiment
    limit = 50 # int | Maximum items to return (optional) (default to 50)

    try:
        # List experiment runs
        api_response = api_instance.experiments_runs_list(experiment_id, limit=limit)
        print("The response of ExperimentsApi->experiments_runs_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->experiments_runs_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique identifier of the experiment | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]

### Return type

[**ExperimentsRunsList200Response**](ExperimentsRunsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of experiment run objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After -  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

