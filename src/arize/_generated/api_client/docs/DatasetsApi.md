# arize._generated.api_client.DatasetsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**datasets_create**](DatasetsApi.md#datasets_create) | **POST** /v2/datasets | Create a dataset
[**datasets_delete**](DatasetsApi.md#datasets_delete) | **DELETE** /v2/datasets/{dataset_id} | Delete a dataset
[**datasets_examples_annotate**](DatasetsApi.md#datasets_examples_annotate) | **POST** /v2/datasets/{dataset_id}/examples/annotate | Annotate a batch of dataset examples
[**datasets_examples_insert**](DatasetsApi.md#datasets_examples_insert) | **POST** /v2/datasets/{dataset_id}/examples | Add new examples to a dataset
[**datasets_examples_list**](DatasetsApi.md#datasets_examples_list) | **GET** /v2/datasets/{dataset_id}/examples | List dataset examples
[**datasets_examples_update**](DatasetsApi.md#datasets_examples_update) | **PATCH** /v2/datasets/{dataset_id}/examples | Update existing examples in a dataset
[**datasets_get**](DatasetsApi.md#datasets_get) | **GET** /v2/datasets/{dataset_id} | Get a dataset
[**datasets_list**](DatasetsApi.md#datasets_list) | **GET** /v2/datasets | List datasets


# **datasets_create**
> Dataset datasets_create(datasets_create_request)

Create a dataset

Create a new dataset with JSON examples. Empty datasets are not allowed.

**Payload Requirements**
- The dataset name must be unique within the given space.
- Each item in `examples[]` may contain **any user-defined fields**.
- Do not include system-managed fields on input: `id`, `created_at`, `updated_at`.
Requests that contain these fields in any example **will be rejected**.
- Each example **must contain at least one property** (i.e., `{}` is invalid).

**Valid example** (create)
```json
{
  "name": "my-dataset",
  "space_id": "spc_123",
  "examples": [
    {
      "question": "What is 2+2?",
      "answer": "4",
      "topic": "math"
    },
    {
      "question": "What is the capital of Spain?",
      "answer": "Madrid",
      "topic": "geography"
    },
  ]
}
```

**Invalid example** ('id' not allowed on create)
```json
{
  "name": "my-dataset",
  "space_id": "spc_123",
  "examples": [
    {
      "id": "ex_1",
      "input": "Hello"
    }
  ]
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.dataset import Dataset
from arize._generated.api_client.models.datasets_create_request import DatasetsCreateRequest
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    datasets_create_request = {"name":"Math Questions Dataset","space_id":"space_12345","examples":[{"question":"What is 2 + 2?","answer":"4","topic":"arithmetic"},{"question":"What is the square root of 16?","answer":"4","topic":"geometry"},{"question":"If 3x = 12, what is x?","answer":"4","topic":"algebra"}]} # DatasetsCreateRequest | Body containing dataset creation parameters

    try:
        # Create a dataset
        api_response = api_instance.datasets_create(datasets_create_request)
        print("The response of DatasetsApi->datasets_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **datasets_create_request** | [**DatasetsCreateRequest**](DatasetsCreateRequest.md)| Body containing dataset creation parameters | 

### Return type

[**Dataset**](Dataset.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | A dataset object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_delete**
> datasets_delete(dataset_id)

Delete a dataset

Delete a dataset by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique identifier of the dataset

    try:
        # Delete a dataset
        api_instance.datasets_delete(dataset_id)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique identifier of the dataset | 

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
**204** | Dataset successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_examples_annotate**
> datasets_examples_annotate(dataset_id, annotate_dataset_examples_request_body)

Annotate a batch of dataset examples

Write human annotations to a batch of examples in a dataset.

**Idempotency**: Writes use upsert semantics — submitting the same annotation
config name for the same example overwrites the previous value. Retrying on
network failure will not create duplicates.

**202 Accepted**: The annotations have been accepted and will be written.
Visibility in read queries may lag by a short interval. No response body
is returned.

**Unmatched record IDs**: If a `record_id` does not correspond to an existing
example in the dataset, the annotation for that record is silently ignored.
No error is returned.

**Payload Requirements**
- `dataset_id` is the path parameter for the target dataset.
- `annotations` is a list of per-example annotation inputs, each identified by `record_id`.
- Annotation names must match existing annotation configs in the dataset's space.
- Up to 1000 examples may be annotated per request.

**Valid example**
```json
{
  "annotations": [
    {"record_id": "ex_abc", "values": [{"name": "quality", "score": 0.8}]}
  ]
}
```

**Invalid example** (annotation name not found in space)
```json
{
  "annotations": [
    {"record_id": "ex_abc", "values": [{"name": "nonexistent_config"}]}
  ]
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotate_dataset_examples_request_body import AnnotateDatasetExamplesRequestBody
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique identifier of the dataset
    annotate_dataset_examples_request_body = {"annotations":[{"record_id":"ex_abc","values":[{"name":"quality","score":0.8}]}]} # AnnotateDatasetExamplesRequestBody | Body containing dataset example annotation batch

    try:
        # Annotate a batch of dataset examples
        api_instance.datasets_examples_annotate(dataset_id, annotate_dataset_examples_request_body)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_examples_annotate: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique identifier of the dataset | 
 **annotate_dataset_examples_request_body** | [**AnnotateDatasetExamplesRequestBody**](AnnotateDatasetExamplesRequestBody.md)| Body containing dataset example annotation batch | 

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
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_examples_insert**
> DatasetVersionWithExampleIds datasets_examples_insert(dataset_id, datasets_examples_insert_request, dataset_version_id=dataset_version_id)

Add new examples to a dataset

Appends new examples to an existing dataset.

If the dataset version is not passed, the latest version is selected.
The inserted examples will be assigned autogenerated, unique IDs.

**Payload Requirements**
- Each item in `examples[]` may contain any user-defined fields.
- Do not include system-managed fields on input: `id`, `created_at`,
`updated_at`. Requests that contain these fields in any example will
be rejected.
- Each example must contain at least one property (i.e., `{}` is invalid).

**Valid example** (create)
```json
{
  "examples": [
    { "question": "What is 2+2?",
      "answer": "4",
      "topic": "math"
    }
  ]
}
```

**Invalid example** ('id' not allowed on create)
```json
{
  "examples": [
    {
      "id": "ex_1",
      "input": "Hello"
    }
  ]
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.dataset_version_with_example_ids import DatasetVersionWithExampleIds
from arize._generated.api_client.models.datasets_examples_insert_request import DatasetsExamplesInsertRequest
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique identifier of the dataset
    datasets_examples_insert_request = {"examples":[{"question":"What is 2 + 2?","answer":"4","topic":"arithmetic"},{"question":"What is the square root of 16?","answer":"4","topic":"geometry"},{"question":"If 3x = 12, what is x?","answer":"4","topic":"algebra"}]} # DatasetsExamplesInsertRequest | Body containing dataset examples for insert (append) operation with auto-generated IDs
    dataset_version_id = 'RGF0YXNldFZlcnNpb246MTIzNDU=' # str | The unique identifier of the dataset version (optional)

    try:
        # Add new examples to a dataset
        api_response = api_instance.datasets_examples_insert(dataset_id, datasets_examples_insert_request, dataset_version_id=dataset_version_id)
        print("The response of DatasetsApi->datasets_examples_insert:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_examples_insert: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique identifier of the dataset | 
 **datasets_examples_insert_request** | [**DatasetsExamplesInsertRequest**](DatasetsExamplesInsertRequest.md)| Body containing dataset examples for insert (append) operation with auto-generated IDs | 
 **dataset_version_id** | **str**| The unique identifier of the dataset version | [optional] 

### Return type

[**DatasetVersionWithExampleIds**](DatasetVersionWithExampleIds.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Examples successfully added to the dataset. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_examples_list**
> DatasetsExamplesList200Response datasets_examples_list(dataset_id, dataset_version_id=dataset_version_id, limit=limit)

List dataset examples

List examples for a given dataset and version.

If version is not passed, the latest version is selected. Examples are
sorted by insertion order.

**Human annotations**: returned in the structured `annotations` array on
each example. Each entry includes `name`, optional `label` / `score` /
`text` / `updated_at`, and an `annotator` (id + email) for per-user
annotations.

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
from arize._generated.api_client.models.datasets_examples_list200_response import DatasetsExamplesList200Response
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique identifier of the dataset
    dataset_version_id = 'RGF0YXNldFZlcnNpb246MTIzNDU=' # str | The unique identifier of the dataset version (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)

    try:
        # List dataset examples
        api_response = api_instance.datasets_examples_list(dataset_id, dataset_version_id=dataset_version_id, limit=limit)
        print("The response of DatasetsApi->datasets_examples_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_examples_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique identifier of the dataset | 
 **dataset_version_id** | **str**| The unique identifier of the dataset version | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]

### Return type

[**DatasetsExamplesList200Response**](DatasetsExamplesList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of dataset examples as structured objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_examples_update**
> DatasetVersionWithExampleIds datasets_examples_update(dataset_id, datasets_examples_update_request, dataset_version_id=dataset_version_id)

Update existing examples in a dataset

Updates existing dataset examples by matching their `id` field.

an example ID does not match any existing example in the dataset
version, it will be ignored. In other words, only examples with IDs
that already exist will be updated. To add new examples, use the
Insert Dataset Examples endpoint.

Adding columns that do not exist in the dataset schema is allowed, but
removing existing columns is not.

Optionally, the update can create a new version of the dataset. In
this case, the outcome of the update will be reflected only in the new
version, while the previous version remains unchanged. If a new
version is not created, the updates will be applied directly (in place)
to the specified version.

**Payload Requirements**
- Each item in `examples[]` may contain any user-defined fields.
- Each item in `examples[]` must include the `id` field to identify the
example to update.
- Do not include system-managed fields on input: `created_at`, `updated_at`.
Requests that contain these fields in any example will be rejected.
- Each example must contain at least one property (i.e., `{}` is invalid).

**Valid example** (create)
```json
{
  "examples": [
    {
      "id": "ex_001",
      "question": "What is 2+2?",
      "answer": "4",
      "topic": "math"
    }
  ]
}
```

**Invalid example** ('id' missing for update)
```json
{
  "examples": [
    {
      "input": "Hello"
    }
  ]
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.dataset_version_with_example_ids import DatasetVersionWithExampleIds
from arize._generated.api_client.models.datasets_examples_update_request import DatasetsExamplesUpdateRequest
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique identifier of the dataset
    datasets_examples_update_request = {"examples":[{"id":"example_001","question":"What is 2 * 2?"},{"id":"example_002","question":"What is the square root of 64?","answer":"8"},{"id":"example_003","question":"If 9x = 36, what is x?","topic":"algebra"}]} # DatasetsExamplesUpdateRequest | Body containing dataset examples for update operation by ID matching
    dataset_version_id = 'RGF0YXNldFZlcnNpb246MTIzNDU=' # str | The unique identifier of the dataset version (optional)

    try:
        # Update existing examples in a dataset
        api_response = api_instance.datasets_examples_update(dataset_id, datasets_examples_update_request, dataset_version_id=dataset_version_id)
        print("The response of DatasetsApi->datasets_examples_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_examples_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique identifier of the dataset | 
 **datasets_examples_update_request** | [**DatasetsExamplesUpdateRequest**](DatasetsExamplesUpdateRequest.md)| Body containing dataset examples for update operation by ID matching | 
 **dataset_version_id** | **str**| The unique identifier of the dataset version | [optional] 

### Return type

[**DatasetVersionWithExampleIds**](DatasetVersionWithExampleIds.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Examples successfully updated in the dataset. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_get**
> Dataset datasets_get(dataset_id)

Get a dataset

Get a dataset object by its ID.

This includes the dataset's versions, sorted by creation date, with
the most recently-created version first.

This endpoint does not include the dataset's examples. To get the examples
of a specific dataset version, use the List Dataset Examples endpoint.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.dataset import Dataset
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique identifier of the dataset

    try:
        # Get a dataset
        api_response = api_instance.datasets_get(dataset_id)
        print("The response of DatasetsApi->datasets_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique identifier of the dataset | 

### Return type

[**Dataset**](Dataset.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A dataset object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **datasets_list**
> DatasetsList200Response datasets_list(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List datasets

List datasets the user has access to.

The datasets are sorted by creation date, with the most recently created
datasets coming first.

The dataset versions are not included in this response. To get the
versions of a specific dataset, use the Get Dataset by ID endpoint.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.datasets_list200_response import DatasetsList200Response
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
    api_instance = arize._generated.api_client.DatasetsApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List datasets
        api_response = api_instance.datasets_list(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of DatasetsApi->datasets_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->datasets_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**DatasetsList200Response**](DatasetsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of dataset objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

