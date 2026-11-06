# arize._generated.api_client.AlignmentApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_alignment_run**](AlignmentApi.md#create_alignment_run) | **POST** /v2/alignment-tasks/{alignment_task_id}/runs | Create alignment run
[**create_alignment_task**](AlignmentApi.md#create_alignment_task) | **POST** /v2/alignment-tasks | Create alignment task
[**get_alignment_task**](AlignmentApi.md#get_alignment_task) | **GET** /v2/alignment-tasks/{alignment_task_id} | Get alignment task
[**list_alignment_runs**](AlignmentApi.md#list_alignment_runs) | **GET** /v2/alignment-tasks/{alignment_task_id}/runs | List alignment runs
[**list_alignment_tasks**](AlignmentApi.md#list_alignment_tasks) | **GET** /v2/alignment-tasks | List alignment tasks


# **create_alignment_run**
> AlignmentRun create_alignment_run(alignment_task_id, create_alignment_run_request)

Create alignment run

Scores how closely one evaluator version's labels matched the human
annotations on a dataset version, using the column pair and comparison
function named by the alignment task in the path.

Scoring is synchronous: the score is computed before the response is
returned, so the created run already carries its `score` and
`comparable_row_count`. There is no run to poll for.

**A run that could not be scored still returns `201`.** The run resource was
created, so it is returned with `status: FAILED` and a public
`failed_reason`. Backend error details are kept in server logs.
A failed score is not a server error and is not reported as `5xx`. Check
`status` on every response rather than relying on the status code alone.

Read `score` and `comparable_row_count` together. `score` is `null`, not
`0`, when no row had both a judge label and a human annotation: nothing was
comparable, so no agreement was measured. A `score` of `0` means rows were
compared and none agreed. A high `score` over a small
`comparable_row_count` is not a meaningful result.

A finished run is never rewritten. Scoring the same version again creates
another run, because both the annotations and the task's columns can change
between scores.

**Payload Requirements**
- `evaluator_version_id` is required: the evaluator version to score. The
  caller must have update access to the evaluator that owns it.
- `dataset_version_id` is required and must belong to the alignment task's
  dataset. A dataset version from another dataset returns `404`, the same as
  one that does not exist.
- `predicted_entity` is required and names what produced the judge labels:
  `entity_type: DATASET_VERSION` when the labels were already on the
  dataset, or `entity_type: EXPERIMENT` when an experiment produced them.
  An entity that does not belong to the same dataset returns `404`.
- System-managed fields (`id`, `created_at`, `updated_at`, `status`,
  `score`, `comparable_row_count`, `failed_reason`) are rejected on input.
  `alignment_task_id` is taken from the path and is rejected in the body.

The response records `predicted_entity` on successful and failed runs.
Malformed IDs, incorrect field types, and a body that is not an object
return `400`. Missing required values, unsupported entity types, and
unknown fields return `422`.

**Valid example**
```json
{
  "evaluator_version_id": "RXZhbHVhdG9yVmVyc2lvbjoxMDpYeVp3",
  "dataset_version_id": "RGF0YXNldFZlcnNpb246NTY6UXJTdA==",
  "predicted_entity": {
    "id": "RGF0YXNldFZlcnNpb246NTY6UXJTdA==",
    "entity_type": "DATASET_VERSION"
  }
}
```

**Invalid example** (`score` is computed by the server and cannot be supplied)
```json
{
  "evaluator_version_id": "RXZhbHVhdG9yVmVyc2lvbjoxMDpYeVp3",
  "dataset_version_id": "RGF0YXNldFZlcnNpb246NTY6UXJTdA==",
  "predicted_entity": {
    "id": "RGF0YXNldFZlcnNpb246NTY6UXJTdA==",
    "entity_type": "DATASET_VERSION"
  },
  "score": 0.9
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.alignment_run import AlignmentRun
from arize._generated.api_client.models.create_alignment_run_request import CreateAlignmentRunRequest
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
    api_instance = arize._generated.api_client.AlignmentApi(api_client)
    alignment_task_id = 'T25saW5lVGFzazoxMjphQmNE' # str | The unique alignment task identifier (base64)
    create_alignment_run_request = {"evaluator_version_id":"RXZhbHVhdG9yVmVyc2lvbjoxMDpYeVp3","dataset_version_id":"RGF0YXNldFZlcnNpb246NTY6UXJTdA==","predicted_entity":{"id":"RGF0YXNldFZlcnNpb246NTY6UXJTdA==","entity_type":"DATASET_VERSION"}} # CreateAlignmentRunRequest | Body containing alignment run parameters.

    try:
        # Create alignment run
        api_response = api_instance.create_alignment_run(alignment_task_id, create_alignment_run_request)
        print("The response of AlignmentApi->create_alignment_run:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AlignmentApi->create_alignment_run: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **alignment_task_id** | **str**| The unique alignment task identifier (base64) | 
 **create_alignment_run_request** | [**CreateAlignmentRunRequest**](CreateAlignmentRunRequest.md)| Body containing alignment run parameters. | 

### Return type

[**AlignmentRun**](AlignmentRun.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created alignment run |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_alignment_task**
> AlignmentTask create_alignment_task(create_alignment_task_request)

Create alignment task

Creates a task that compares an evaluator label column with a human
annotation column on a dataset. Create an alignment run to calculate a
score with this task.

Requests with the same `dataset_id`, `judge_label_column`,
`annotation_column`, and `alignment_function` return the same task,
including concurrent requests. An omitted `alignment_function` is
equivalent to `EXACT_MATCH`. The task name is not part of this key.

Returns `201` when a task is created, or `200` when an existing task is
returned. Reuse does not change the task name or timestamps.

**Payload Requirements**
- `dataset_id` identifies the dataset to compare.
- `name`, `judge_label_column`, and `annotation_column` must each contain
  1 to 255 characters and must not contain only whitespace.
- `judge_label_column` identifies the evaluator labels. Use
  `eval.<name>.label` for labels on a dataset, or `output.label` for
  experiment output.
- `annotation_column` identifies the human annotations.
- `alignment_function` defaults to `EXACT_MATCH`, which compares labels
  for equality.
- Columns are checked when a run is scored, not when a task is created.
  A column can be populated after task creation. A scoring failure is
  recorded on the run.
- Unknown fields, including `id`, `created_at`, and `updated_at`, are
  rejected.

Malformed IDs, incorrect field types, and a body that is not an object
return `400`. Missing required values, unsupported comparison functions,
unknown fields, and values outside the length limits return `422`.
A dataset that does not exist or that the caller cannot read returns
`404`. A dataset the caller can read but lacks permission to update
returns `403`.

**Valid example**
```json
{
  "dataset_id": "RGF0YXNldDoxMjM0NQ==",
  "name": "Hallucination judge alignment",
  "judge_label_column": "eval.hallucination.label",
  "annotation_column": "annotation.hallucination.label",
  "alignment_function": "EXACT_MATCH"
}
```

**Invalid example** (system-managed field supplied on input)
```json
{
  "id": "T25saW5lVGFzazoxMjphQmNE",
  "dataset_id": "RGF0YXNldDoxMjM0NQ==",
  "name": "Hallucination judge alignment",
  "judge_label_column": "eval.hallucination.label",
  "annotation_column": "annotation.hallucination.label"
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.alignment_task import AlignmentTask
from arize._generated.api_client.models.create_alignment_task_request import CreateAlignmentTaskRequest
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
    api_instance = arize._generated.api_client.AlignmentApi(api_client)
    create_alignment_task_request = {"dataset_id":"RGF0YXNldDoxMjM0NQ==","name":"Hallucination judge alignment","judge_label_column":"eval.hallucination.label","annotation_column":"annotation.hallucination.label","alignment_function":"EXACT_MATCH"} # CreateAlignmentTaskRequest | Parameters for creating an alignment task.

    try:
        # Create alignment task
        api_response = api_instance.create_alignment_task(create_alignment_task_request)
        print("The response of AlignmentApi->create_alignment_task:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AlignmentApi->create_alignment_task: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_alignment_task_request** | [**CreateAlignmentTaskRequest**](CreateAlignmentTaskRequest.md)| Parameters for creating an alignment task. | 

### Return type

[**AlignmentTask**](AlignmentTask.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the existing task with the same dataset, columns, and comparison function |  -  |
**201** | Returns the new alignment task |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_alignment_task**
> AlignmentTask get_alignment_task(alignment_task_id)

Get alignment task

Returns one alignment task by its ID, including the column pair and
comparison function it compares with.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.alignment_task import AlignmentTask
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
    api_instance = arize._generated.api_client.AlignmentApi(api_client)
    alignment_task_id = 'T25saW5lVGFzazoxMjphQmNE' # str | The unique alignment task identifier (base64)

    try:
        # Get alignment task
        api_response = api_instance.get_alignment_task(alignment_task_id)
        print("The response of AlignmentApi->get_alignment_task:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AlignmentApi->get_alignment_task: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **alignment_task_id** | **str**| The unique alignment task identifier (base64) | 

### Return type

[**AlignmentTask**](AlignmentTask.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the new alignment task |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_alignment_runs**
> ListAlignmentRunsResponse list_alignment_runs(alignment_task_id, evaluator_id=evaluator_id, limit=limit, cursor=cursor)

List alignment runs

Lists the alignment runs recorded against one alignment task, most recently
created first.

Runs within one task are directly comparable: they share a dataset, a
column pair and a comparison function, so the only thing that varies is
which evaluator version was scored. Scores from *different* tasks are not
comparable with each other -- a different column pair is a different scale.

Use `evaluator_id` to narrow the list to one evaluator's versions, which is
how you see whether that evaluator's agreement with human labels improved as
it was edited. Keep `evaluator_id` identical on every request in a
pagination sequence: a cursor records only a position in the result order,
so pairing one with a different filter silently returns a page of the new
filter's results rather than reporting an error.

Read `score` and `comparable_row_count` together on every run. A `score` of
`null` means no row had both a judge label and a human annotation, which is
not the same as a `score` of `0`; and a `score` over a small
`comparable_row_count` is not a meaningful result. Runs with
`status: FAILED` carry a `null` score and a `failed_reason`.

A task with no runs returns an empty list with `200`. A `404` means the
task does not exist or is not visible to the caller -- the two are
deliberately indistinguishable.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_alignment_runs_response import ListAlignmentRunsResponse
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
    api_instance = arize._generated.api_client.AlignmentApi(api_client)
    alignment_task_id = 'T25saW5lVGFzazoxMjphQmNE' # str | The unique alignment task identifier (base64)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | Limit the results to runs that scored a version of this evaluator. Omit to return runs for every evaluator measured against the task.  (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List alignment runs
        api_response = api_instance.list_alignment_runs(alignment_task_id, evaluator_id=evaluator_id, limit=limit, cursor=cursor)
        print("The response of AlignmentApi->list_alignment_runs:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AlignmentApi->list_alignment_runs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **alignment_task_id** | **str**| The unique alignment task identifier (base64) | 
 **evaluator_id** | **str**| Limit the results to runs that scored a version of this evaluator. Omit to return runs for every evaluator measured against the task.  | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListAlignmentRunsResponse**](ListAlignmentRunsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of alignment runs |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_alignment_tasks**
> ListAlignmentTasksResponse list_alignment_tasks(dataset_id, limit=limit, cursor=cursor)

List alignment tasks

Lists the alignment tasks defined on one dataset, most recently created
first.

Use this to find a task's `id` before scoring an evaluator against it. A
dataset with no alignment tasks returns an empty list with `200`. A `404`
means the dataset does not exist or is not visible to the caller -- the two
are deliberately indistinguishable.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_alignment_tasks_response import ListAlignmentTasksResponse
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
    api_instance = arize._generated.api_client.AlignmentApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The dataset to list for (base64 identifier (base64))
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List alignment tasks
        api_response = api_instance.list_alignment_tasks(dataset_id, limit=limit, cursor=cursor)
        print("The response of AlignmentApi->list_alignment_tasks:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AlignmentApi->list_alignment_tasks: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The dataset to list for (base64 identifier (base64)) | 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListAlignmentTasksResponse**](ListAlignmentTasksResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of alignment task objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

