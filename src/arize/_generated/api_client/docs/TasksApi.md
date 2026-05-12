# arize._generated.api_client.TasksApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**task_runs_cancel**](TasksApi.md#task_runs_cancel) | **POST** /v2/task-runs/{run_id}/cancel | Cancel task run
[**task_runs_get**](TasksApi.md#task_runs_get) | **GET** /v2/task-runs/{run_id} | Get task run
[**tasks_create**](TasksApi.md#tasks_create) | **POST** /v2/tasks | Create task
[**tasks_delete**](TasksApi.md#tasks_delete) | **DELETE** /v2/tasks/{task_id} | Delete task
[**tasks_get**](TasksApi.md#tasks_get) | **GET** /v2/tasks/{task_id} | Get task
[**tasks_list**](TasksApi.md#tasks_list) | **GET** /v2/tasks | List tasks
[**tasks_list_runs**](TasksApi.md#tasks_list_runs) | **GET** /v2/tasks/{task_id}/runs | List task runs
[**tasks_trigger_run**](TasksApi.md#tasks_trigger_run) | **POST** /v2/tasks/{task_id}/trigger | Trigger a task run
[**tasks_update**](TasksApi.md#tasks_update) | **PATCH** /v2/tasks/{task_id} | Update task


# **task_runs_cancel**
> TaskRun task_runs_cancel(run_id)

Cancel task run

Cancel a running task run. Only valid when the run's current status
is `pending` or `running`.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task_run import TaskRun
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    run_id = 'VGFza1J1bjoxMjM0NQ==' # str | The task run global ID (base64)

    try:
        # Cancel task run
        api_response = api_instance.task_runs_cancel(run_id)
        print("The response of TasksApi->task_runs_cancel:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->task_runs_cancel: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| The task run global ID (base64) | 

### Return type

[**TaskRun**](TaskRun.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a single task run object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **task_runs_get**
> TaskRun task_runs_get(run_id)

Get task run

Returns a single task run. Use this to poll for status updates.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task_run import TaskRun
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    run_id = 'VGFza1J1bjoxMjM0NQ==' # str | The task run global ID (base64)

    try:
        # Get task run
        api_response = api_instance.task_runs_get(run_id)
        print("The response of TasksApi->task_runs_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->task_runs_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| The task run global ID (base64) | 

### Return type

[**TaskRun**](TaskRun.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a single task run object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_create**
> Task tasks_create(tasks_create_request)

Create task

Creates a new task. Supported task types:

| `type` | Data source | Notes |
|---|---|---|
| `template_evaluation` | `project_id` or `dataset_id` | Requires `evaluators`. Supports continuous operation. |
| `code_evaluation` | `project_id` or `dataset_id` | Requires `evaluators`. Supports continuous operation. |
| `run_experiment` | `dataset_id` only | Requires `run_configuration`. Never continuous. |

For `run_experiment` tasks the run configuration is stored on the task.
Each trigger (`POST /v2/tasks/{task_id}/trigger`) supplies per-run fields
(`experiment_name`, optional example subset, etc.) and starts an async run.
Poll `GET /v2/task-runs/{run_id}` until `status` reaches a terminal state.

**Validation Rules (template_evaluation / code_evaluation)**
- At least one evaluator is required.
- Duplicate evaluator IDs are not allowed.
- When `dataset_id` is provided, `experiment_ids` must contain at least one entry.
- `sampling_rate` and `is_continuous` are only supported on project-based tasks.

**Validation Rules (run_experiment)**
- `dataset_id` is required; `project_id` must be omitted.
- `run_configuration` is required; `evaluators`, `experiment_ids`, `sampling_rate`,
  `is_continuous`, and `query_filter` must be omitted.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task import Task
from arize._generated.api_client.models.tasks_create_request import TasksCreateRequest
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    tasks_create_request = {"name":"Production Hallucination Check","type":"template_evaluation","project_id":"TW9kZWw6MTIzOmFCY0Q=","sampling_rate":1,"is_continuous":true,"query_filter":"metadata.environment = 'production'","evaluators":[{"evaluator_id":"RXZhbHVhdG9yOjEyOmFCY0Q=","column_mappings":{"input":"attributes.input.value","output":"attributes.output.value"}}]} # TasksCreateRequest | Body containing task creation parameters. The `type` field is the discriminator.  | `type` | Schema | |---|---| | `template_evaluation` | `CreateTemplateEvaluationTaskRequest` | | `code_evaluation` | `CreateCodeEvaluationTaskRequest` | | `run_experiment` | `CreateRunExperimentTaskRequest` |  `run_experiment` tasks do not run continuously — they must be triggered explicitly via `POST /v2/tasks/{task_id}/trigger` each time.  For `template_evaluation` / `code_evaluation` tasks, exactly one of `project_id` or `dataset_id` must be provided. When `dataset_id` is provided, `experiment_ids` must contain at least one entry. `is_continuous` and `sampling_rate` are only supported for project-based tasks. 

    try:
        # Create task
        api_response = api_instance.tasks_create(tasks_create_request)
        print("The response of TasksApi->tasks_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **tasks_create_request** | [**TasksCreateRequest**](TasksCreateRequest.md)| Body containing task creation parameters. The &#x60;type&#x60; field is the discriminator.  | &#x60;type&#x60; | Schema | |---|---| | &#x60;template_evaluation&#x60; | &#x60;CreateTemplateEvaluationTaskRequest&#x60; | | &#x60;code_evaluation&#x60; | &#x60;CreateCodeEvaluationTaskRequest&#x60; | | &#x60;run_experiment&#x60; | &#x60;CreateRunExperimentTaskRequest&#x60; |  &#x60;run_experiment&#x60; tasks do not run continuously — they must be triggered explicitly via &#x60;POST /v2/tasks/{task_id}/trigger&#x60; each time.  For &#x60;template_evaluation&#x60; / &#x60;code_evaluation&#x60; tasks, exactly one of &#x60;project_id&#x60; or &#x60;dataset_id&#x60; must be provided. When &#x60;dataset_id&#x60; is provided, &#x60;experiment_ids&#x60; must contain at least one entry. &#x60;is_continuous&#x60; and &#x60;sampling_rate&#x60; are only supported for project-based tasks.  | 

### Return type

[**Task**](Task.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created task |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Invalid request |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_delete**
> tasks_delete(task_id)

Delete task

Deletes a task and all its associated resources (evaluator configs, runs, etc.).
This operation is irreversible.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    task_id = 'VGFzazoxMjM0NQ==' # str | The task global ID (base64)

    try:
        # Delete task
        api_instance.tasks_delete(task_id)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The task global ID (base64) | 

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
**204** | Task deleted successfully |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_get**
> Task tasks_get(task_id)

Get task

Returns a single task by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task import Task
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    task_id = 'VGFzazoxMjM0NQ==' # str | The task global ID (base64)

    try:
        # Get task
        api_response = api_instance.tasks_get(task_id)
        print("The response of TasksApi->tasks_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The task global ID (base64) | 

### Return type

[**Task**](Task.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a single task object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_list**
> TasksList200Response tasks_list(space_id=space_id, space_name=space_name, name=name, project_id=project_id, dataset_id=dataset_id, type=type, limit=limit, cursor=cursor)

List tasks

List tasks the user has access to, with cursor-based pagination.

Filter by space, space name, task name, project, dataset, or task type using query parameters.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.tasks_list200_response import TasksList200Response
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    project_id = 'UHJvamVjdDoxMjM0NQ==' # str | Filter to tasks for a specific project (base64 global ID) (optional)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | Filter to a specific dataset (base64 global ID) (optional)
    type = 'template_evaluation' # str | Filter by task type: template_evaluation, code_evaluation, or run_experiment (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List tasks
        api_response = api_instance.tasks_list(space_id=space_id, space_name=space_name, name=name, project_id=project_id, dataset_id=dataset_id, type=type, limit=limit, cursor=cursor)
        print("The response of TasksApi->tasks_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **project_id** | **str**| Filter to tasks for a specific project (base64 global ID) | [optional] 
 **dataset_id** | **str**| Filter to a specific dataset (base64 global ID) | [optional] 
 **type** | **str**| Filter by task type: template_evaluation, code_evaluation, or run_experiment | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**TasksList200Response**](TasksList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of task objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_list_runs**
> TasksListRuns200Response tasks_list_runs(task_id, status=status, limit=limit, cursor=cursor)

List task runs

List all runs for a task with cursor-based pagination.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.tasks_list_runs200_response import TasksListRuns200Response
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    task_id = 'VGFzazoxMjM0NQ==' # str | The task global ID (base64)
    status = 'completed' # str | Filter by run status: pending, running, completed, failed, cancelled (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List task runs
        api_response = api_instance.tasks_list_runs(task_id, status=status, limit=limit, cursor=cursor)
        print("The response of TasksApi->tasks_list_runs:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_list_runs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The task global ID (base64) | 
 **status** | **str**| Filter by run status: pending, running, completed, failed, cancelled | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**TasksListRuns200Response**](TasksListRuns200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of task run objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_trigger_run**
> TaskRun tasks_trigger_run(task_id, tasks_trigger_run_request=tasks_trigger_run_request)

Trigger a task run

Triggers a new run on an existing task. The run is queued and processed
asynchronously. Poll `GET /v2/task-runs/{run_id}` until the run reaches a
terminal status (`completed`, `failed`, or `cancelled`).

**For `run_experiment` tasks**

Supply `experiment_name` (required) plus any of the optional per-run fields:
`dataset_version_id`, `example_ids` (exclusive with `max_examples`),
`max_examples`, `tracing_metadata`, `evaluation_task_ids`.

The fields `data_start_time`, `data_end_time`, `max_spans`,
`override_evaluations`, and `experiment_ids` are not applicable and will
return 400 if supplied.

The response includes `experiment_id` once the experiment is provisioned.

**For `template_evaluation` / `code_evaluation` tasks**

Supply `data_start_time`, `data_end_time`, `max_spans`,
`override_evaluations`, and/or `experiment_ids` as needed.
`run_experiment`-specific fields are not applicable for these task types.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task_run import TaskRun
from arize._generated.api_client.models.tasks_trigger_run_request import TasksTriggerRunRequest
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    task_id = 'VGFzazoxMjM0NQ==' # str | The task global ID (base64)
    tasks_trigger_run_request = {"data_start_time":"2026-03-01T00:00:00Z","data_end_time":"2026-03-07T00:00:00Z","max_spans":5000,"override_evaluations":false} # TasksTriggerRunRequest | Trigger body for `POST /v2/tasks/{task_id}/trigger`. The server derives the task type from the URL's task record and selects the appropriate schema; the body itself does not carry a `task_type` field.  | Task type | Schema | |---|---| | `template_evaluation` | `TriggerEvaluationTaskRunRequest` | | `code_evaluation` | `TriggerEvaluationTaskRunRequest` | | `run_experiment` | `TriggerRunExperimentTaskRunRequest` |  Sending a field that is not valid for the resolved task type returns 400. For `template_evaluation` and `code_evaluation` tasks all trigger fields are optional — an empty body is valid and uses server defaults.  (optional)

    try:
        # Trigger a task run
        api_response = api_instance.tasks_trigger_run(task_id, tasks_trigger_run_request=tasks_trigger_run_request)
        print("The response of TasksApi->tasks_trigger_run:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_trigger_run: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The task global ID (base64) | 
 **tasks_trigger_run_request** | [**TasksTriggerRunRequest**](TasksTriggerRunRequest.md)| Trigger body for &#x60;POST /v2/tasks/{task_id}/trigger&#x60;. The server derives the task type from the URL&#39;s task record and selects the appropriate schema; the body itself does not carry a &#x60;task_type&#x60; field.  | Task type | Schema | |---|---| | &#x60;template_evaluation&#x60; | &#x60;TriggerEvaluationTaskRunRequest&#x60; | | &#x60;code_evaluation&#x60; | &#x60;TriggerEvaluationTaskRunRequest&#x60; | | &#x60;run_experiment&#x60; | &#x60;TriggerRunExperimentTaskRunRequest&#x60; |  Sending a field that is not valid for the resolved task type returns 400. For &#x60;template_evaluation&#x60; and &#x60;code_evaluation&#x60; tasks all trigger fields are optional — an empty body is valid and uses server defaults.  | [optional] 

### Return type

[**TaskRun**](TaskRun.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created task run |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **tasks_update**
> Task tasks_update(task_id, tasks_update_request)

Update task

Update a task's mutable fields. At least one field must be provided.
Omitted fields are left unchanged.

When `evaluators` is provided, the entire evaluator list is replaced.

`sampling_rate` and `is_continuous` are only applicable for project-based tasks.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task import Task
from arize._generated.api_client.models.tasks_update_request import TasksUpdateRequest
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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    task_id = 'VGFzazoxMjM0NQ==' # str | The task global ID (base64)
    tasks_update_request = {"name":"Updated Task Name","sampling_rate":0.5,"query_filter":"metadata.environment = 'staging'"} # TasksUpdateRequest | PATCH body for `PATCH /v2/tasks/{task_id}`. The server derives the task type from the URL's task record and selects the appropriate schema; the body itself does not carry a `type` field.  | Task type | Schema | |---|---| | `template_evaluation` | `UpdateEvaluationTaskRequest` | | `code_evaluation` | `UpdateEvaluationTaskRequest` | | `run_experiment` | `UpdateRunExperimentTaskRequest` |  For `template_evaluation` and `code_evaluation` tasks, at least one of `name`, `sampling_rate`, `is_continuous`, `query_filter`, or `evaluators` must be provided.  For `run_experiment` tasks, at least one of `name` or `run_configuration` must be provided. When `run_configuration` is provided the stored config is atomically replaced.  Sending a field that is not valid for the resolved task type returns 400 (e.g. `evaluators` on a `run_experiment` task, or `run_configuration` on an evaluation task). 

    try:
        # Update task
        api_response = api_instance.tasks_update(task_id, tasks_update_request)
        print("The response of TasksApi->tasks_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->tasks_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The task global ID (base64) | 
 **tasks_update_request** | [**TasksUpdateRequest**](TasksUpdateRequest.md)| PATCH body for &#x60;PATCH /v2/tasks/{task_id}&#x60;. The server derives the task type from the URL&#39;s task record and selects the appropriate schema; the body itself does not carry a &#x60;type&#x60; field.  | Task type | Schema | |---|---| | &#x60;template_evaluation&#x60; | &#x60;UpdateEvaluationTaskRequest&#x60; | | &#x60;code_evaluation&#x60; | &#x60;UpdateEvaluationTaskRequest&#x60; | | &#x60;run_experiment&#x60; | &#x60;UpdateRunExperimentTaskRequest&#x60; |  For &#x60;template_evaluation&#x60; and &#x60;code_evaluation&#x60; tasks, at least one of &#x60;name&#x60;, &#x60;sampling_rate&#x60;, &#x60;is_continuous&#x60;, &#x60;query_filter&#x60;, or &#x60;evaluators&#x60; must be provided.  For &#x60;run_experiment&#x60; tasks, at least one of &#x60;name&#x60; or &#x60;run_configuration&#x60; must be provided. When &#x60;run_configuration&#x60; is provided the stored config is atomically replaced.  Sending a field that is not valid for the resolved task type returns 400 (e.g. &#x60;evaluators&#x60; on a &#x60;run_experiment&#x60; task, or &#x60;run_configuration&#x60; on an evaluation task).  | 

### Return type

[**Task**](Task.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the updated task |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

