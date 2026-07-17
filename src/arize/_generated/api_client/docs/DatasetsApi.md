# arize._generated.api_client.DatasetsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**annotate_dataset_examples**](DatasetsApi.md#annotate_dataset_examples) | **POST** /v2/datasets/{dataset_id}/examples/annotate | Annotate a batch of dataset examples
[**create_dataset**](DatasetsApi.md#create_dataset) | **POST** /v2/datasets | Create a dataset
[**delete_dataset**](DatasetsApi.md#delete_dataset) | **DELETE** /v2/datasets/{dataset_id} | Delete a dataset
[**delete_dataset_examples**](DatasetsApi.md#delete_dataset_examples) | **DELETE** /v2/datasets/{dataset_id}/examples | Delete dataset examples
[**get_dataset**](DatasetsApi.md#get_dataset) | **GET** /v2/datasets/{dataset_id} | Get a dataset
[**insert_dataset_examples**](DatasetsApi.md#insert_dataset_examples) | **POST** /v2/datasets/{dataset_id}/examples | Add new examples to a dataset
[**list_dataset_examples**](DatasetsApi.md#list_dataset_examples) | **GET** /v2/datasets/{dataset_id}/examples | List dataset examples
[**list_datasets**](DatasetsApi.md#list_datasets) | **GET** /v2/datasets | List datasets
[**update_dataset**](DatasetsApi.md#update_dataset) | **PATCH** /v2/datasets/{dataset_id} | Update a dataset
[**update_dataset_examples**](DatasetsApi.md#update_dataset_examples) | **PATCH** /v2/datasets/{dataset_id}/examples | Update existing examples in a dataset


# **annotate_dataset_examples**
> annotate_dataset_examples(dataset_id, annotate_dataset_examples_request)

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

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotate_dataset_examples_request import AnnotateDatasetExamplesRequest
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)
    annotate_dataset_examples_request = {"annotations":[{"record_id":"ex_abc","values":[{"name":"quality","score":0.8}]}]} # AnnotateDatasetExamplesRequest | Body containing dataset example annotation batch

    try:
        # Annotate a batch of dataset examples
        api_instance.annotate_dataset_examples(dataset_id, annotate_dataset_examples_request)
    except Exception as e:
        print("Exception when calling DatasetsApi->annotate_dataset_examples: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 
 **annotate_dataset_examples_request** | [**AnnotateDatasetExamplesRequest**](AnnotateDatasetExamplesRequest.md)| Body containing dataset example annotation batch | 

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

# **create_dataset**
> Dataset create_dataset(create_dataset_request)

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
from arize._generated.api_client.models.create_dataset_request import CreateDatasetRequest
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
    create_dataset_request = {"name":"Math Questions Dataset","space_id":"space_12345","examples":[{"question":"What is 2 + 2?","answer":"4","topic":"arithmetic"},{"question":"What is the square root of 16?","answer":"4","topic":"geometry"},{"question":"If 3x = 12, what is x?","answer":"4","topic":"algebra"}]} # CreateDatasetRequest | Body containing dataset creation parameters

    try:
        # Create a dataset
        api_response = api_instance.create_dataset(create_dataset_request)
        print("The response of DatasetsApi->create_dataset:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->create_dataset: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_dataset_request** | [**CreateDatasetRequest**](CreateDatasetRequest.md)| Body containing dataset creation parameters | 

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
**422** | Unprocessable entity |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_dataset**
> delete_dataset(dataset_id)

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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)

    try:
        # Delete a dataset
        api_instance.delete_dataset(dataset_id)
    except Exception as e:
        print("Exception when calling DatasetsApi->delete_dataset: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 

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

# **delete_dataset_examples**
> DeleteDatasetExamplesResponse delete_dataset_examples(dataset_id, delete_dataset_examples_request)

Delete dataset examples

Delete a collection of examples from a dataset by their IDs.

The delete is partial-tolerant: examples that exist in the selected version
are deleted, and every requested ID that was not deleted is reported back.

A `200 OK` response always includes:
- `completed` — `true` if the operation finished and no retry is needed;
  `false` if it could not fully complete (retry the full request).
- `deleted_example_ids` — example IDs confirmed deleted in this request.
- `not_deleted_example_ids` — requested IDs not deleted: either not found in
  the selected version (never added, or already deleted), or not completed
  when `completed` is `false`.

The delete operation is idempotent — re-submitting already-deleted IDs is safe.

**Payload Requirements**
- `dataset_version_id` is required and identifies the version to delete from.
- `example_ids` must contain at least one ID and at most 1000 IDs.
- `example_ids` must not contain duplicate or empty IDs.

**Valid example**
```json
{
  "dataset_version_id": "RGF0YXNldFZlcnNpb246MTIzNDU=",
  "example_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
  ]
}
```

**Invalid example** (missing `dataset_version_id`)
```json
{
  "example_ids": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

  <Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.delete_dataset_examples_request import DeleteDatasetExamplesRequest
from arize._generated.api_client.models.delete_dataset_examples_response import DeleteDatasetExamplesResponse
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)
    delete_dataset_examples_request = {"dataset_version_id":"RGF0YXNldFZlcnNpb246MTIzNDU=","example_ids":["550e8400-e29b-41d4-a716-446655440000","6ba7b810-9dad-11d1-80b4-00c04fd430c8"]} # DeleteDatasetExamplesRequest | Body containing the IDs of dataset examples to delete

    try:
        # Delete dataset examples
        api_response = api_instance.delete_dataset_examples(dataset_id, delete_dataset_examples_request)
        print("The response of DatasetsApi->delete_dataset_examples:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->delete_dataset_examples: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 
 **delete_dataset_examples_request** | [**DeleteDatasetExamplesRequest**](DeleteDatasetExamplesRequest.md)| Body containing the IDs of dataset examples to delete | 

### Return type

[**DeleteDatasetExamplesResponse**](DeleteDatasetExamplesResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Dataset examples deleted. The delete is partial-tolerant: existing examples are deleted and every requested ID not deleted is reported back. The response body always includes: - &#x60;completed&#x60;: &#x60;true&#x60; if the operation finished; &#x60;false&#x60; if it could not fully   complete (retry the full request). - &#x60;deleted_example_ids&#x60;: IDs confirmed deleted. - &#x60;not_deleted_example_ids&#x60;: requested IDs not deleted — not found in the   selected version, or not completed when &#x60;completed&#x60; is &#x60;false&#x60;.  |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |
**503** | Returned as &#x60;503 Service Unavailable&#x60; when the request fails after partially completing. &#x60;deleted_example_ids&#x60; lists what was already deleted and &#x60;not_deleted_example_ids&#x60; lists what still needs deletion. The caller should retry the original full request — the delete operation is idempotent.  |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_dataset**
> Dataset get_dataset(dataset_id)

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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)

    try:
        # Get a dataset
        api_response = api_instance.get_dataset(dataset_id)
        print("The response of DatasetsApi->get_dataset:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->get_dataset: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **insert_dataset_examples**
> DatasetVersionWithExampleIds insert_dataset_examples(dataset_id, insert_dataset_examples_request, dataset_version_id=dataset_version_id)

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
from arize._generated.api_client.models.insert_dataset_examples_request import InsertDatasetExamplesRequest
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)
    insert_dataset_examples_request = {"examples":[{"question":"What is 2 + 2?","answer":"4","topic":"arithmetic"},{"question":"What is the square root of 16?","answer":"4","topic":"geometry"},{"question":"If 3x = 12, what is x?","answer":"4","topic":"algebra"}]} # InsertDatasetExamplesRequest | Body containing dataset examples for insert (append) operation with auto-generated IDs
    dataset_version_id = 'RGF0YXNldFZlcnNpb246MTIzNDU=' # str | The unique identifier of the dataset version (optional)

    try:
        # Add new examples to a dataset
        api_response = api_instance.insert_dataset_examples(dataset_id, insert_dataset_examples_request, dataset_version_id=dataset_version_id)
        print("The response of DatasetsApi->insert_dataset_examples:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->insert_dataset_examples: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 
 **insert_dataset_examples_request** | [**InsertDatasetExamplesRequest**](InsertDatasetExamplesRequest.md)| Body containing dataset examples for insert (append) operation with auto-generated IDs | 
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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_dataset_examples**
> ListDatasetExamplesResponse list_dataset_examples(dataset_id, dataset_version_id=dataset_version_id, limit=limit, cursor=cursor)

List dataset examples

List examples for a given dataset and version.

If version is not passed, the latest version is selected. Examples are
returned in ascending order of `created_at`, with `id` as a tiebreaker.
This order is stable across pages, so cursor pagination never skips or
repeats an example.

**Human annotations**: returned in the structured `annotations` array on
each example. Each entry includes `name`, optional `label` / `score` /
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
from arize._generated.api_client.models.list_dataset_examples_response import ListDatasetExamplesResponse
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)
    dataset_version_id = 'RGF0YXNldFZlcnNpb246MTIzNDU=' # str | The unique identifier of the dataset version (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List dataset examples
        api_response = api_instance.list_dataset_examples(dataset_id, dataset_version_id=dataset_version_id, limit=limit, cursor=cursor)
        print("The response of DatasetsApi->list_dataset_examples:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->list_dataset_examples: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 
 **dataset_version_id** | **str**| The unique identifier of the dataset version | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListDatasetExamplesResponse**](ListDatasetExamplesResponse.md)

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

# **list_datasets**
> ListDatasetsResponse list_datasets(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

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
from arize._generated.api_client.models.list_datasets_response import ListDatasetsResponse
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
        api_response = api_instance.list_datasets(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of DatasetsApi->list_datasets:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->list_datasets: %s\n" % e)
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

[**ListDatasetsResponse**](ListDatasetsResponse.md)

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_dataset**
> Dataset update_dataset(dataset_id, update_dataset_request)

Update a dataset

Update an existing dataset by its ID.

**Payload Requirements**
- `name` is required.
- `name` must be unique within the space (409 Conflict if duplicate).
- `name` cannot be empty or whitespace-only.

**Valid example**
```json
{
  "name": "Updated Dataset Name"
}
```

**Invalid example** (empty body — no fields provided)
```json
{}
```

  <Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.dataset import Dataset
from arize._generated.api_client.models.update_dataset_request import UpdateDatasetRequest
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)
    update_dataset_request = {"name":"Updated Dataset Name"} # UpdateDatasetRequest | Body containing dataset update parameters.

    try:
        # Update a dataset
        api_response = api_instance.update_dataset(dataset_id, update_dataset_request)
        print("The response of DatasetsApi->update_dataset:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->update_dataset: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 
 **update_dataset_request** | [**UpdateDatasetRequest**](UpdateDatasetRequest.md)| Body containing dataset update parameters. | 

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
**200** | A dataset object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_dataset_examples**
> DatasetVersionWithExampleIds update_dataset_examples(dataset_id, update_dataset_examples_request, dataset_version_id=dataset_version_id)

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
from arize._generated.api_client.models.update_dataset_examples_request import UpdateDatasetExamplesRequest
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
    dataset_id = 'RGF0YXNldDoxMjM0NQ==' # str | The unique dataset identifier (base64)
    update_dataset_examples_request = {"examples":[{"id":"example_001","question":"What is 2 * 2?"},{"id":"example_002","question":"What is the square root of 64?","answer":"8"},{"id":"example_003","question":"If 9x = 36, what is x?","topic":"algebra"}]} # UpdateDatasetExamplesRequest | Body containing dataset examples for update operation by ID matching
    dataset_version_id = 'RGF0YXNldFZlcnNpb246MTIzNDU=' # str | The unique identifier of the dataset version (optional)

    try:
        # Update existing examples in a dataset
        api_response = api_instance.update_dataset_examples(dataset_id, update_dataset_examples_request, dataset_version_id=dataset_version_id)
        print("The response of DatasetsApi->update_dataset_examples:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DatasetsApi->update_dataset_examples: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **dataset_id** | **str**| The unique dataset identifier (base64) | 
 **update_dataset_examples_request** | [**UpdateDatasetExamplesRequest**](UpdateDatasetExamplesRequest.md)| Body containing dataset examples for update operation by ID matching | 
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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

