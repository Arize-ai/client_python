# arize._generated.api_client.TasksApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**cancel_task_run**](TasksApi.md#cancel_task_run) | **POST** /v2/task-runs/{run_id}/cancel | Cancel task run
[**create_task**](TasksApi.md#create_task) | **POST** /v2/tasks | Create task
[**delete_task**](TasksApi.md#delete_task) | **DELETE** /v2/tasks/{task_id} | Delete task
[**get_task**](TasksApi.md#get_task) | **GET** /v2/tasks/{task_id} | Get task
[**get_task_run**](TasksApi.md#get_task_run) | **GET** /v2/task-runs/{run_id} | Get task run
[**list_task_runs**](TasksApi.md#list_task_runs) | **GET** /v2/tasks/{task_id}/runs | List task runs
[**list_tasks**](TasksApi.md#list_tasks) | **GET** /v2/tasks | List tasks
[**trigger_task_run**](TasksApi.md#trigger_task_run) | **POST** /v2/tasks/{task_id}/trigger | Trigger a task run
[**update_task**](TasksApi.md#update_task) | **PATCH** /v2/tasks/{task_id} | Update task


# **cancel_task_run**
> TaskRun cancel_task_run(run_id)

Cancel task run

Cancel a running task run. Only valid when the run's current status
is `pending` or `running`.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    run_id = 'VGFza1J1bjoxMjM0NQ==' # str | The unique task run identifier (base64)

    try:
        # Cancel task run
        api_response = api_instance.cancel_task_run(run_id)
        print("The response of TasksApi->cancel_task_run:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->cancel_task_run: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| The unique task run identifier (base64) | 

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

# **create_task**
> Task create_task(create_task_request)

Create task

Creates a new task. Supported task types:

| `type` | Data source | Notes |
|---|---|---|
| `TEMPLATE_EVALUATION` | `project_id` or `dataset_id` | Requires `evaluators`. Supports continuous operation. |
| `CODE_EVALUATION` | `project_id` or `dataset_id` | Requires `evaluators`. Supports continuous operation. |
| `RUN_EXPERIMENT` | `dataset_id` only | Requires `run_configuration`. Never continuous. |

For `RUN_EXPERIMENT` tasks the run configuration is stored on the task.
Each trigger (`POST /v2/tasks/{task_id}/trigger`) supplies per-run fields
(`experiment_name`, optional example subset, etc.) and starts an async run.
Poll `GET /v2/task-runs/{run_id}` until `status` reaches a terminal state.

**Payload Requirements (template_evaluation / code_evaluation)**
- At least one evaluator is required.
- Duplicate evaluator IDs are not allowed.
- When `dataset_id` is provided, `experiment_ids` must contain at least one entry.
- `sampling_rate` and `is_continuous` are only supported on project-based tasks.
- System-managed fields (`id`, `created_at`, `updated_at`) are rejected on input.

**Payload Requirements (run_experiment)**
- `dataset_id` is required; `project_id` must be omitted.
- `run_configuration` is required; `evaluators`, `experiment_ids`, `sampling_rate`,
  `is_continuous`, and `query_filter` must be omitted.

**Valid example** (template_evaluation, project-based)
```json
{
  "name": "Production Hallucination Check",
  "type": "TEMPLATE_EVALUATION",
  "project_id": "TW9kZWw6MTIzOmFCY0Q=",
  "sampling_rate": 1.0,
  "is_continuous": true,
  "evaluators": [
    {
      "evaluator_id": "RXZhbHVhdG9yOjEyOmFCY0Q=",
      "column_mappings": {"input": "attributes.input.value", "output": "attributes.output.value"}
    }
  ]
}
```

**Invalid example** (run_experiment missing `run_configuration`)
```json
{
  "name": "My Experiment",
  "type": "RUN_EXPERIMENT",
  "dataset_id": "RGF0YXNldDo1NjpxUndY"
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_task_request import CreateTaskRequest
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
    create_task_request = {"name":"Production Hallucination Check","type":"TEMPLATE_EVALUATION","project_id":"TW9kZWw6MTIzOmFCY0Q=","sampling_rate":1,"is_continuous":true,"query_filter":"metadata.environment = 'production'","evaluators":[{"evaluator_id":"RXZhbHVhdG9yOjEyOmFCY0Q=","column_mappings":{"input":"attributes.input.value","output":"attributes.output.value"}}]} # CreateTaskRequest | Body containing task creation parameters. The `type` field is the discriminator.  | `type` | Schema | |---|---| | `TEMPLATE_EVALUATION` | `CreateTemplateEvaluationTaskRequest` | | `CODE_EVALUATION` | `CreateCodeEvaluationTaskRequest` | | `RUN_EXPERIMENT` | `CreateRunExperimentTaskRequest` |  `RUN_EXPERIMENT` tasks do not run continuously — they must be triggered explicitly via `POST /v2/tasks/{task_id}/trigger` each time.  For `TEMPLATE_EVALUATION` / `CODE_EVALUATION` tasks, exactly one of `project_id` or `dataset_id` must be provided. When `dataset_id` is provided, `experiment_ids` must contain at least one entry. `is_continuous` and `sampling_rate` are only supported for project-based tasks. 

    try:
        # Create task
        api_response = api_instance.create_task(create_task_request)
        print("The response of TasksApi->create_task:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->create_task: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_task_request** | [**CreateTaskRequest**](CreateTaskRequest.md)| Body containing task creation parameters. The &#x60;type&#x60; field is the discriminator.  | &#x60;type&#x60; | Schema | |---|---| | &#x60;TEMPLATE_EVALUATION&#x60; | &#x60;CreateTemplateEvaluationTaskRequest&#x60; | | &#x60;CODE_EVALUATION&#x60; | &#x60;CreateCodeEvaluationTaskRequest&#x60; | | &#x60;RUN_EXPERIMENT&#x60; | &#x60;CreateRunExperimentTaskRequest&#x60; |  &#x60;RUN_EXPERIMENT&#x60; tasks do not run continuously — they must be triggered explicitly via &#x60;POST /v2/tasks/{task_id}/trigger&#x60; each time.  For &#x60;TEMPLATE_EVALUATION&#x60; / &#x60;CODE_EVALUATION&#x60; tasks, exactly one of &#x60;project_id&#x60; or &#x60;dataset_id&#x60; must be provided. When &#x60;dataset_id&#x60; is provided, &#x60;experiment_ids&#x60; must contain at least one entry. &#x60;is_continuous&#x60; and &#x60;sampling_rate&#x60; are only supported for project-based tasks.  | 

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
**201** | Returns a single task object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_task**
> delete_task(task_id)

Delete task

Deletes a task and all its associated resources (evaluator configs, runs, etc.).
This operation is irreversible.

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
    api_instance = arize._generated.api_client.TasksApi(api_client)
    task_id = 'VGFzazoxMjM0NQ==' # str | The unique task identifier (base64)

    try:
        # Delete task
        api_instance.delete_task(task_id)
    except Exception as e:
        print("Exception when calling TasksApi->delete_task: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The unique task identifier (base64) | 

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

# **get_task**
> Task get_task(task_id)

Get task

Returns a single task by its ID.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    task_id = 'VGFzazoxMjM0NQ==' # str | The unique task identifier (base64)

    try:
        # Get task
        api_response = api_instance.get_task(task_id)
        print("The response of TasksApi->get_task:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->get_task: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The unique task identifier (base64) | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_task_run**
> TaskRun get_task_run(run_id)

Get task run

Returns a single task run. Use this to poll for status updates.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    run_id = 'VGFza1J1bjoxMjM0NQ==' # str | The unique task run identifier (base64)

    try:
        # Get task run
        api_response = api_instance.get_task_run(run_id)
        print("The response of TasksApi->get_task_run:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->get_task_run: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **run_id** | **str**| The unique task run identifier (base64) | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_task_runs**
> ListTaskRunsResponse list_task_runs(task_id, status=status, limit=limit, cursor=cursor)

List task runs

List all runs for a task with cursor-based pagination.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_task_runs_response import ListTaskRunsResponse
from arize._generated.api_client.models.task_run_status import TaskRunStatus
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
    task_id = 'VGFzazoxMjM0NQ==' # str | The unique task identifier (base64)
    status = arize._generated.api_client.TaskRunStatus() # TaskRunStatus | Filter by run status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List task runs
        api_response = api_instance.list_task_runs(task_id, status=status, limit=limit, cursor=cursor)
        print("The response of TasksApi->list_task_runs:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->list_task_runs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The unique task identifier (base64) | 
 **status** | [**TaskRunStatus**](.md)| Filter by run status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListTaskRunsResponse**](ListTaskRunsResponse.md)

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

# **list_tasks**
> ListTasksResponse list_tasks(space_id=space_id, space_name=space_name, name=name, project_id=project_id, dataset_id=dataset_id, type=type, limit=limit, cursor=cursor)

List tasks

List tasks the user has access to, with cursor-based pagination.

Filter by space, space name, task name, project, dataset, or task type using query parameters.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_tasks_response import ListTasksResponse
from arize._generated.api_client.models.task_type import TaskType
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
    project_id = 'UHJvamVjdDoxMjM0NQ==' # str | Filter to tasks for a specific project (base64 identifier (base64)) (optional)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | Filter to a specific dataset (base64 identifier (base64)) (optional)
    type = arize._generated.api_client.TaskType() # TaskType | Filter by task type: TEMPLATE_EVALUATION, CODE_EVALUATION, or RUN_EXPERIMENT (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List tasks
        api_response = api_instance.list_tasks(space_id=space_id, space_name=space_name, name=name, project_id=project_id, dataset_id=dataset_id, type=type, limit=limit, cursor=cursor)
        print("The response of TasksApi->list_tasks:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->list_tasks: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **project_id** | **str**| Filter to tasks for a specific project (base64 identifier (base64)) | [optional] 
 **dataset_id** | **str**| Filter to a specific dataset (base64 identifier (base64)) | [optional] 
 **type** | [**TaskType**](.md)| Filter by task type: TEMPLATE_EVALUATION, CODE_EVALUATION, or RUN_EXPERIMENT | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListTasksResponse**](ListTasksResponse.md)

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

# **trigger_task_run**
> TaskRun trigger_task_run(task_id, trigger_task_run_request=trigger_task_run_request)

Trigger a task run

Triggers a new run on an existing task. The run is queued and processed
asynchronously. Poll `GET /v2/task-runs/{run_id}` until the run reaches a
terminal status (`completed`, `failed`, or `cancelled`).

**Payload Requirements**
- Fields must match the task's type; sending inapplicable fields returns 400.
- For `TEMPLATE_EVALUATION` / `CODE_EVALUATION` tasks, all trigger fields are optional — an empty body uses server defaults.
- For `RUN_EXPERIMENT` tasks, `experiment_name` is required.

**For `RUN_EXPERIMENT` tasks**

Supply `experiment_name` (required) plus any of the optional per-run fields:
`dataset_version_id`, `example_ids` (exclusive with `max_examples`),
`max_examples`, `tracing_metadata`, `evaluation_task_ids`.

The fields `data_start_time`, `data_end_time`, `max_spans`,
`override_evaluations`, and `experiment_ids` are not applicable and will
return 400 if supplied.

The response includes `experiment_id` once the experiment is provisioned.

**For `TEMPLATE_EVALUATION` / `CODE_EVALUATION` tasks**

Supply `data_start_time`, `data_end_time`, `max_spans`,
`override_evaluations`, and/or `experiment_ids` as needed.
`RUN_EXPERIMENT`-specific fields are not applicable for these task types.

**Valid example** (trigger a run_experiment run)
```json
{
  "experiment_name": "GPT-4o Baseline v2",
  "max_examples": 50
}
```

**Invalid example** (run_experiment trigger missing required `experiment_name`)
```json
{
  "max_examples": 50
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task_run import TaskRun
from arize._generated.api_client.models.trigger_task_run_request import TriggerTaskRunRequest
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
    task_id = 'VGFzazoxMjM0NQ==' # str | The unique task identifier (base64)
    trigger_task_run_request = {"data_start_time":"2026-03-01T00:00:00Z","data_end_time":"2026-03-07T00:00:00Z","max_spans":5000,"override_evaluations":false} # TriggerTaskRunRequest | Trigger body for `POST /v2/tasks/{task_id}/trigger`. The server derives the task type from the URL's task record and selects the appropriate schema; the body itself does not carry a `task_type` field.  | Task type | Schema | |---|---| | `TEMPLATE_EVALUATION` | `TriggerEvaluationTaskRunRequest` | | `CODE_EVALUATION` | `TriggerEvaluationTaskRunRequest` | | `RUN_EXPERIMENT` | `TriggerRunExperimentTaskRunRequest` |  Sending a field that is not valid for the resolved task type returns 400. For `TEMPLATE_EVALUATION` and `CODE_EVALUATION` tasks all trigger fields are optional — an empty body is valid and uses server defaults.  (optional)

    try:
        # Trigger a task run
        api_response = api_instance.trigger_task_run(task_id, trigger_task_run_request=trigger_task_run_request)
        print("The response of TasksApi->trigger_task_run:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->trigger_task_run: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The unique task identifier (base64) | 
 **trigger_task_run_request** | [**TriggerTaskRunRequest**](TriggerTaskRunRequest.md)| Trigger body for &#x60;POST /v2/tasks/{task_id}/trigger&#x60;. The server derives the task type from the URL&#39;s task record and selects the appropriate schema; the body itself does not carry a &#x60;task_type&#x60; field.  | Task type | Schema | |---|---| | &#x60;TEMPLATE_EVALUATION&#x60; | &#x60;TriggerEvaluationTaskRunRequest&#x60; | | &#x60;CODE_EVALUATION&#x60; | &#x60;TriggerEvaluationTaskRunRequest&#x60; | | &#x60;RUN_EXPERIMENT&#x60; | &#x60;TriggerRunExperimentTaskRunRequest&#x60; |  Sending a field that is not valid for the resolved task type returns 400. For &#x60;TEMPLATE_EVALUATION&#x60; and &#x60;CODE_EVALUATION&#x60; tasks all trigger fields are optional — an empty body is valid and uses server defaults.  | [optional] 

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
**201** | Returns a single task run object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_task**
> Task update_task(task_id, update_task_request)

Update task

Update a task's mutable fields. At least one field must be provided.
Omitted fields are left unchanged.

**Payload Requirements**
- At least one mutable field must be provided.
- When `evaluators` is provided, the entire evaluator list is replaced.
- `sampling_rate` and `is_continuous` are only applicable for project-based tasks.
- Fields not valid for the task's type return 400 (e.g. `run_configuration` on an evaluation task).
- System-managed fields (`id`, `type`, `created_at`, `updated_at`) cannot be modified.

**Valid example** (update evaluation task)
```json
{
  "name": "Updated Hallucination Check",
  "sampling_rate": 0.5,
  "query_filter": "metadata.environment = 'staging'"
}
```

**Invalid example** (no fields provided)
```json
{}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.task import Task
from arize._generated.api_client.models.update_task_request import UpdateTaskRequest
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
    task_id = 'VGFzazoxMjM0NQ==' # str | The unique task identifier (base64)
    update_task_request = {"name":"Updated Task Name","sampling_rate":0.5,"query_filter":"metadata.environment = 'staging'"} # UpdateTaskRequest | PATCH body for `PATCH /v2/tasks/{task_id}`. The server derives the task type from the URL's task record and selects the appropriate schema; the body itself does not carry a `type` field.  | Task type | Schema | |---|---| | `TEMPLATE_EVALUATION` | `UpdateEvaluationTaskRequest` | | `CODE_EVALUATION` | `UpdateEvaluationTaskRequest` | | `RUN_EXPERIMENT` | `UpdateRunExperimentTaskRequest` |  For `TEMPLATE_EVALUATION` and `CODE_EVALUATION` tasks, at least one of `name`, `sampling_rate`, `is_continuous`, `query_filter`, or `evaluators` must be provided.  For `RUN_EXPERIMENT` tasks, at least one of `name` or `run_configuration` must be provided. When `run_configuration` is provided the stored config is atomically replaced.  Sending a field that is not valid for the resolved task type returns 400 (e.g. `evaluators` on a `RUN_EXPERIMENT` task, or `run_configuration` on an evaluation task). 

    try:
        # Update task
        api_response = api_instance.update_task(task_id, update_task_request)
        print("The response of TasksApi->update_task:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TasksApi->update_task: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **task_id** | **str**| The unique task identifier (base64) | 
 **update_task_request** | [**UpdateTaskRequest**](UpdateTaskRequest.md)| PATCH body for &#x60;PATCH /v2/tasks/{task_id}&#x60;. The server derives the task type from the URL&#39;s task record and selects the appropriate schema; the body itself does not carry a &#x60;type&#x60; field.  | Task type | Schema | |---|---| | &#x60;TEMPLATE_EVALUATION&#x60; | &#x60;UpdateEvaluationTaskRequest&#x60; | | &#x60;CODE_EVALUATION&#x60; | &#x60;UpdateEvaluationTaskRequest&#x60; | | &#x60;RUN_EXPERIMENT&#x60; | &#x60;UpdateRunExperimentTaskRequest&#x60; |  For &#x60;TEMPLATE_EVALUATION&#x60; and &#x60;CODE_EVALUATION&#x60; tasks, at least one of &#x60;name&#x60;, &#x60;sampling_rate&#x60;, &#x60;is_continuous&#x60;, &#x60;query_filter&#x60;, or &#x60;evaluators&#x60; must be provided.  For &#x60;RUN_EXPERIMENT&#x60; tasks, at least one of &#x60;name&#x60; or &#x60;run_configuration&#x60; must be provided. When &#x60;run_configuration&#x60; is provided the stored config is atomically replaced.  Sending a field that is not valid for the resolved task type returns 400 (e.g. &#x60;evaluators&#x60; on a &#x60;RUN_EXPERIMENT&#x60; task, or &#x60;run_configuration&#x60; on an evaluation task).  | 

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
**200** | Returns a single task object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

