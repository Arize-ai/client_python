# arize._generated.api_client.ExperimentsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**add_experiment_tags**](ExperimentsApi.md#add_experiment_tags) | **POST** /v2/experiments/{experiment_id}/tags | Attach tags to a experiment
[**annotate_experiment_runs**](ExperimentsApi.md#annotate_experiment_runs) | **POST** /v2/experiments/{experiment_id}/runs/annotate | Annotate a batch of experiment runs
[**create_experiment**](ExperimentsApi.md#create_experiment) | **POST** /v2/experiments | Create an experiment
[**delete_experiment**](ExperimentsApi.md#delete_experiment) | **DELETE** /v2/experiments/{experiment_id} | Delete an experiment
[**get_experiment**](ExperimentsApi.md#get_experiment) | **GET** /v2/experiments/{experiment_id} | Get an experiment
[**insert_experiment_runs**](ExperimentsApi.md#insert_experiment_runs) | **POST** /v2/experiments/{experiment_id}/runs | Append runs to an experiment
[**list_experiment_runs**](ExperimentsApi.md#list_experiment_runs) | **GET** /v2/experiments/{experiment_id}/runs | List experiment runs
[**list_experiment_tags**](ExperimentsApi.md#list_experiment_tags) | **GET** /v2/experiments/{experiment_id}/tags | List tags on an experiment
[**list_experiments**](ExperimentsApi.md#list_experiments) | **GET** /v2/experiments | List experiments


# **add_experiment_tags**
> ListTagsResponse add_experiment_tags(experiment_id, add_tags_request)

Attach tags to a experiment

Attach one or more existing tags to a experiment.

**Payload Requirements**
- `tag_ids` is required and must contain between 1 and 100 tag IDs.
- Every tag must already exist and belong to the same space as the
  experiment. A tag from another space returns `422`.
- Attaching a tag that is already attached is idempotent, so the same
  request can be retried safely.
- Unrecognized fields are rejected with `422` rather than ignored.

Returns `200` with the experiment's complete tag list, not `201`: attaching
an existing tag creates no new resource.

**Valid example**
```json
{
  "tag_ids": ["VGFnOjEyMzQ1", "VGFnOjEyMzQ2"]
}
```

**Invalid example** (empty list)
```json
{
  "tag_ids": []
}
```
```json
{
  "type": "https://arize.com/docs/ax/rest-reference/errors#validation-error",
  "title": "Unprocessable Entity",
  "status": 422,
  "detail": "tag_ids must contain at least 1 tag ID",
  "request_id": "req_01HZY6X8E7"
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.add_tags_request import AddTagsRequest
from arize._generated.api_client.models.list_tags_response import ListTagsResponse
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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)
    add_tags_request = {"tag_ids":["VGFnOjEyMzQ1","VGFnOjEyMzQ2"]} # AddTagsRequest | Body containing the IDs of the tags to attach to the resource

    try:
        # Attach tags to a experiment
        api_response = api_instance.add_experiment_tags(experiment_id, add_tags_request)
        print("The response of ExperimentsApi->add_experiment_tags:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->add_experiment_tags: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 
 **add_tags_request** | [**AddTagsRequest**](AddTagsRequest.md)| Body containing the IDs of the tags to attach to the resource | 

### Return type

[**ListTagsResponse**](ListTagsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the tags attached to the resource |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotate_experiment_runs**
> annotate_experiment_runs(experiment_id, annotate_experiment_runs_request)

Annotate a batch of experiment runs

Write human annotations to a batch of runs in an experiment.

**Idempotency**: Writes use upsert semantics — submitting the same annotation
config name for the same run overwrites the previous value. Retrying on
network failure will not create duplicates.

**202 Accepted**: The annotations have been accepted and will be written.
Visibility in read queries may lag by a short interval. No response body
is returned.

**Unmatched record IDs**: If a `record_id` does not correspond to an existing
run in the experiment, the annotation for that record is silently ignored.
No error is returned.

**Payload Requirements**
- `experiment_id` is the path parameter for the target experiment.
- `annotations` is a list of per-run annotation inputs, each identified by `record_id`.
- Annotation names must match existing annotation configs in the experiment's space.
- Up to 1000 runs may be annotated per request.

**Valid example**
```json
{
  "annotations": [
    {"record_id": "run_abc", "values": [{"name": "quality", "label": "good"}]}
  ]
}
```

**Invalid example** (annotation name not found in space)
```json
{
  "annotations": [
    {"record_id": "run_abc", "values": [{"name": "nonexistent_config"}]}
  ]
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotate_experiment_runs_request import AnnotateExperimentRunsRequest
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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)
    annotate_experiment_runs_request = {"annotations":[{"record_id":"run_abc","values":[{"name":"quality","label":"good"}]}]} # AnnotateExperimentRunsRequest | Body containing experiment run annotation batch

    try:
        # Annotate a batch of experiment runs
        api_instance.annotate_experiment_runs(experiment_id, annotate_experiment_runs_request)
    except Exception as e:
        print("Exception when calling ExperimentsApi->annotate_experiment_runs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 
 **annotate_experiment_runs_request** | [**AnnotateExperimentRunsRequest**](AnnotateExperimentRunsRequest.md)| Body containing experiment run annotation batch | 

### Return type

void (empty response body)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**202** | Annotations written successfully. The annotations have been accepted and will be written. Visibility in read queries may lag by a short interval. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_experiment**
> Experiment create_experiment(create_experiment_request)

Create an experiment

Create a new experiment. Empty experiments are not allowed.

An experiment belongs to a space and may optionally be associated with a
dataset.

Experiments are composed of "runs". Each experiment run (JSON object)
must include an `output` field containing the task's output. When the
experiment is associated with a dataset, each run must also include an
`example_id` referencing an example in that dataset.

Payload Requirements
- Provide exactly one of `dataset_id` or `space_id`.
- The `name` must be unique within the dataset it's associated with, or
  within the space when it isn't associated with a dataset, and must not
  contain double quotes (`"`) or backslashes (`\`).
- Provide at least one run in `experiment_runs`.
- Each run must include:
  - `output` -- model/task output for the run
  - `example_id` -- a correlation ID linking this run to a dataset example.
  Required only when the experiment is associated with a dataset; its
  existence in the dataset is never validated.
  - You may include any additional fields per run that can be used for
  analysis or filtering. For example: `model`, `latency_ms`,
  `temperature`, `prompt`, `tool_calls`, etc.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_experiment_request import CreateExperimentRequest
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
    create_experiment_request = {"name":"My Experiment Name","dataset_id":"dataset_12345","experiment_runs":[{"example_id":"example_001","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":118,"prompt":"Answer the math question briefly."},{"example_id":"example_002","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":132},{"example_id":"example_003","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":125}]} # CreateExperimentRequest | Body containing experiment creation parameters

    try:
        # Create an experiment
        api_response = api_instance.create_experiment(create_experiment_request)
        print("The response of ExperimentsApi->create_experiment:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->create_experiment: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_experiment_request** | [**CreateExperimentRequest**](CreateExperimentRequest.md)| Body containing experiment creation parameters | 

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
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_experiment**
> delete_experiment(experiment_id)

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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)

    try:
        # Delete an experiment
        api_instance.delete_experiment(experiment_id)
    except Exception as e:
        print("Exception when calling ExperimentsApi->delete_experiment: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 

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
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_experiment**
> Experiment get_experiment(experiment_id)

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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)

    try:
        # Get an experiment
        api_response = api_instance.get_experiment(experiment_id)
        print("The response of ExperimentsApi->get_experiment:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->get_experiment: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **insert_experiment_runs**
> ExperimentWithRunIds insert_experiment_runs(experiment_id, insert_experiment_runs_request)

Append runs to an experiment

Append new runs to an existing experiment.

**Payload Requirements**
- Provide between 1 and 1000 runs in `experiment_runs`.
- Each run must include:
  - `output` -- model/task output for the run
  - `example_id` -- a correlation ID linking this run to a dataset example.
  Required only when the experiment is associated with a dataset; its
  existence in the dataset is never validated.
  - You may include any additional fields per run that can be used for
  analysis or filtering. For example: `model`, `latency_ms`,
  `temperature`, `prompt`, `tool_calls`, etc.

**Valid example**
```json
{
  "experiment_runs": [
    {"example_id": "example_001", "output": "4", "model": "gpt-4o-mini"}
  ]
}
```

**Invalid example** (missing required output field)
```json
{
  "experiment_runs": [
    {"example_id": "example_001"}
  ]
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.experiment_with_run_ids import ExperimentWithRunIds
from arize._generated.api_client.models.insert_experiment_runs_request import InsertExperimentRunsRequest
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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)
    insert_experiment_runs_request = {"experiment_runs":[{"example_id":"example_001","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":118},{"example_id":"example_002","output":"4","model":"gpt-4o-mini","temperature":0.2,"latency_ms":132}]} # InsertExperimentRunsRequest | Body containing experiment runs to append to the experiment

    try:
        # Append runs to an experiment
        api_response = api_instance.insert_experiment_runs(experiment_id, insert_experiment_runs_request)
        print("The response of ExperimentsApi->insert_experiment_runs:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->insert_experiment_runs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 
 **insert_experiment_runs_request** | [**InsertExperimentRunsRequest**](InsertExperimentRunsRequest.md)| Body containing experiment runs to append to the experiment | 

### Return type

[**ExperimentWithRunIds**](ExperimentWithRunIds.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Experiment with the IDs of the newly inserted runs. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_experiment_runs**
> ListExperimentRunsResponse list_experiment_runs(experiment_id, limit=limit, cursor=cursor)

List experiment runs

List runs for a given experiment.

The runs are returned in a stable insertion order.

**Human annotations**: returned in the structured `annotations` array on
each run. Each entry includes `name`, optional `label` / `score` /
`text` / `updated_at`, and an `annotator` (id + email) for per-user
annotations.

**Pagination**:
- Response includes `pagination` with `has_more` and `next_cursor`.
- Use cursor-based pagination by passing the returned `next_cursor`
value as the `cursor` query parameter in subsequent requests.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_experiment_runs_response import ListExperimentRunsResponse
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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 500. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List experiment runs
        api_response = api_instance.list_experiment_runs(experiment_id, limit=limit, cursor=cursor)
        print("The response of ExperimentsApi->list_experiment_runs:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->list_experiment_runs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 500. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListExperimentRunsResponse**](ListExperimentRunsResponse.md)

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
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_experiment_tags**
> ListTagsResponse list_experiment_tags(experiment_id)

List tags on an experiment

List the tags attached to an experiment.

Tags are shared within the space, so the same tag may appear on many
resources. An experiment with no tags returns an empty list rather than a
404.

Requires read access to the experiment. A caller who cannot read it receives
`404`, identical to the response for an experiment that does not exist.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_tags_response import ListTagsResponse
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
    experiment_id = 'RXhwZXJpbWVudDoxMjM0NQ==' # str | The unique experiment identifier (base64)

    try:
        # List tags on an experiment
        api_response = api_instance.list_experiment_tags(experiment_id)
        print("The response of ExperimentsApi->list_experiment_tags:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->list_experiment_tags: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **experiment_id** | **str**| The unique experiment identifier (base64) | 

### Return type

[**ListTagsResponse**](ListTagsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the tags attached to the resource |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_experiments**
> ListExperimentsResponse list_experiments(dataset_id=dataset_id, space_id=space_id, name=name, limit=limit, cursor=cursor)

List experiments

List experiments a user has access to.

By default, lists every accessible experiment across all spaces the caller
can read, including experiments that are not associated with a dataset.

To narrow the results, provide at most one of:
- `dataset_id` — only experiments run on that dataset.
- `space_id` — only experiments in that space (with or without a dataset).

Providing both `dataset_id` and `space_id` is a validation error.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_experiments_response import ListExperimentsResponse
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | Filter to a specific dataset (base64 identifier (base64)) (optional)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List experiments
        api_response = api_instance.list_experiments(dataset_id=dataset_id, space_id=space_id, name=name, limit=limit, cursor=cursor)
        print("The response of ExperimentsApi->list_experiments:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ExperimentsApi->list_experiments: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| Filter to a specific dataset (base64 identifier (base64)) | [optional] 
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListExperimentsResponse**](ListExperimentsResponse.md)

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

