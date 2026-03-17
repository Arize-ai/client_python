# arize._generated.api_client.AnnotationQueuesApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**annotation_queue_records_list**](AnnotationQueuesApi.md#annotation_queue_records_list) | **GET** /v2/annotation-queues/{annotation_queue_id}/records | List annotation queue records
[**annotation_queues_create**](AnnotationQueuesApi.md#annotation_queues_create) | **POST** /v2/annotation-queues | Create an annotation queue
[**annotation_queues_delete**](AnnotationQueuesApi.md#annotation_queues_delete) | **DELETE** /v2/annotation-queues/{annotation_queue_id} | Delete an annotation queue
[**annotation_queues_get**](AnnotationQueuesApi.md#annotation_queues_get) | **GET** /v2/annotation-queues/{annotation_queue_id} | Get an annotation queue
[**annotation_queues_list**](AnnotationQueuesApi.md#annotation_queues_list) | **GET** /v2/annotation-queues | List annotation queues
[**annotation_queues_records_create**](AnnotationQueuesApi.md#annotation_queues_records_create) | **POST** /v2/annotation-queues/{annotation_queue_id}/records | Create annotation queue records
[**annotation_queues_update**](AnnotationQueuesApi.md#annotation_queues_update) | **PATCH** /v2/annotation-queues/{annotation_queue_id} | Update an annotation queue


# **annotation_queue_records_list**
> AnnotationQueueRecordsList200Response annotation_queue_records_list(annotation_queue_id, cursor=cursor, limit=limit)

List annotation queue records

List the records in an annotation queue with their data and annotations.

Each record includes:
- The record's data as flat key-value pairs
- Any annotations that have been added to the record
- The users assigned to annotate the record and their completion status

**Pagination**:
- Response includes `pagination` with `has_more` and `next_cursor`.
- Use cursor-based pagination by passing the returned `next_cursor`
value as the `cursor` query parameter in subsequent requests.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_queue_records_list200_response import AnnotationQueueRecordsList200Response
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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    annotation_queue_id = 'annotation_queue_id_example' # str | The unique identifier of the annotation queue
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)

    try:
        # List annotation queue records
        api_response = api_instance.annotation_queue_records_list(annotation_queue_id, cursor=cursor, limit=limit)
        print("The response of AnnotationQueuesApi->annotation_queue_records_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queue_records_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_queue_id** | **str**| The unique identifier of the annotation queue | 
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]

### Return type

[**AnnotationQueueRecordsList200Response**](AnnotationQueueRecordsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of annotation queue record objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_queues_create**
> AnnotationQueue annotation_queues_create(create_annotation_queue_request_body)

Create an annotation queue

Create a new annotation queue.

**Payload Requirements**
- The annotation queue name must be unique within the given space (among active queues).
- At least one `annotation_config_id` is required, and all configs must belong to the specified space.
- Do not include system-managed fields on input: `id`, `created_at`, `updated_at`.
- If `assignment_method` is not provided, it defaults to `"all"`.
- If `annotator_emails` are provided, all emails must resolve to existing users with access to the space.

**Valid example**
```json
{
  "name": "Quality Review Queue",
  "space_id": "spc_xyz789",
  "annotation_config_ids": ["ac_abc123"],
  "assignment_method": "all"
}
```

**Valid example with records**
```json
{
  "name": "Quality Review Queue",
  "space_id": "spc_xyz789",
  "annotation_config_ids": ["ac_abc123"],
  "annotator_emails": ["reviewer@example.com"],
  "records": [
    {"record_type": "span", "project_id": "prj_abc", "start_time": "2024-01-15T00:00:00Z", "end_time": "2024-01-16T00:00:00Z", "span_ids": ["span_001"]},
    {"record_type": "example", "dataset_id": "ds_xyz", "example_ids": ["ex_001", "ex_002"]}
  ]
}
```

**Invalid example** (missing required annotation_config_ids)
```json
{
  "name": "My Queue",
  "space_id": "spc_xyz789"
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_queue import AnnotationQueue
from arize._generated.api_client.models.create_annotation_queue_request_body import CreateAnnotationQueueRequestBody
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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    create_annotation_queue_request_body = {"name":"Quality Review Queue","space_id":"spc_xyz789","annotation_config_ids":["ac_abc123"],"assignment_method":"all"} # CreateAnnotationQueueRequestBody | Body containing annotation queue creation parameters

    try:
        # Create an annotation queue
        api_response = api_instance.annotation_queues_create(create_annotation_queue_request_body)
        print("The response of AnnotationQueuesApi->annotation_queues_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queues_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_annotation_queue_request_body** | [**CreateAnnotationQueueRequestBody**](CreateAnnotationQueueRequestBody.md)| Body containing annotation queue creation parameters | 

### Return type

[**AnnotationQueue**](AnnotationQueue.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created annotation queue |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_queues_delete**
> annotation_queues_delete(annotation_queue_id)

Delete an annotation queue

Delete an annotation queue by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    annotation_queue_id = 'annotation_queue_id_example' # str | The unique identifier of the annotation queue

    try:
        # Delete an annotation queue
        api_instance.annotation_queues_delete(annotation_queue_id)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queues_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_queue_id** | **str**| The unique identifier of the annotation queue | 

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
**204** | Annotation queue successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_queues_get**
> AnnotationQueue annotation_queues_get(annotation_queue_id)

Get an annotation queue

Get an annotation queue object by its ID.

This includes the annotation queue's annotation configs, which define the
structure of annotations that can be created in this queue.

This endpoint does not include queue records or annotation progress. To
manage records in a queue, use the Annotation Queue Items endpoints.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_queue import AnnotationQueue
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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    annotation_queue_id = 'annotation_queue_id_example' # str | The unique identifier of the annotation queue

    try:
        # Get an annotation queue
        api_response = api_instance.annotation_queues_get(annotation_queue_id)
        print("The response of AnnotationQueuesApi->annotation_queues_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queues_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_queue_id** | **str**| The unique identifier of the annotation queue | 

### Return type

[**AnnotationQueue**](AnnotationQueue.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An annotation queue object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_queues_list**
> AnnotationQueuesList200Response annotation_queues_list(space_id=space_id, name=name, limit=limit, cursor=cursor)

List annotation queues

List annotation queues the user has access to.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_queues_list200_response import AnnotationQueuesList200Response
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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    space_id = 'space_id_example' # str | Filter search results to a particular space ID (optional)
    name = 'name_example' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. When omitted, no name filter is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List annotation queues
        api_response = api_instance.annotation_queues_list(space_id=space_id, name=name, limit=limit, cursor=cursor)
        print("The response of AnnotationQueuesApi->annotation_queues_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queues_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. When omitted, no name filter is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**AnnotationQueuesList200Response**](AnnotationQueuesList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of annotation queue objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_queues_records_create**
> AnnotationQueuesRecordsCreate201Response annotation_queues_records_create(annotation_queue_id, add_annotation_queue_records_request_body)

Create annotation queue records

Add new records from either spans (a project) or from dataset examples to an existing annotation queue.

**Payload Requirements**
  - At least one record source is required.
  - At most 2 record sources are allowed per request
  - For span record source: `start_time` must be before `end_time`, and the range must not exceed 7 days.
  - For dataset record source: all `example_ids` must be non-empty strings.
  - For spans record source: all `span_ids` must be non-empty strings.
  - At most 500 records total may be added in one request

<Note>If no example_ids are provided for a dataset record source, all examples in the dataset will be added to the queue.</Note>

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.add_annotation_queue_records_request_body import AddAnnotationQueueRecordsRequestBody
from arize._generated.api_client.models.annotation_queues_records_create201_response import AnnotationQueuesRecordsCreate201Response
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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    annotation_queue_id = 'annotation_queue_id_example' # str | The unique identifier of the annotation queue
    add_annotation_queue_records_request_body = arize._generated.api_client.AddAnnotationQueueRecordsRequestBody() # AddAnnotationQueueRecordsRequestBody | Body containing records to add to an annotation queue

    try:
        # Create annotation queue records
        api_response = api_instance.annotation_queues_records_create(annotation_queue_id, add_annotation_queue_records_request_body)
        print("The response of AnnotationQueuesApi->annotation_queues_records_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queues_records_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_queue_id** | **str**| The unique identifier of the annotation queue | 
 **add_annotation_queue_records_request_body** | [**AddAnnotationQueueRecordsRequestBody**](AddAnnotationQueueRecordsRequestBody.md)| Body containing records to add to an annotation queue | 

### Return type

[**AnnotationQueuesRecordsCreate201Response**](AnnotationQueuesRecordsCreate201Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created annotation queue records |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_queues_update**
> AnnotationQueue annotation_queues_update(annotation_queue_id, update_annotation_queue_request_body)

Update an annotation queue

Update an annotation queue by its ID. At least one field must be provided.

**Payload Requirements:**
- At least one of `name`, `instructions`, `annotation_config_ids`, or `annotator_emails` must be provided
- `name` must be unique within the space (409 Conflict if duplicate)
- `annotation_config_ids` replaces all existing config associations; all configs must belong to the same space as the queue
- `annotator_emails` replaces all existing user assignments; all users must have active accounts

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_queue import AnnotationQueue
from arize._generated.api_client.models.update_annotation_queue_request_body import UpdateAnnotationQueueRequestBody
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
    api_instance = arize._generated.api_client.AnnotationQueuesApi(api_client)
    annotation_queue_id = 'annotation_queue_id_example' # str | The unique identifier of the annotation queue
    update_annotation_queue_request_body = {"name":"Updated Queue Name","instructions":"Review each response for accuracy and helpfulness","annotation_config_ids":["ac_abc123"],"annotator_emails":["reviewer@example.com"]} # UpdateAnnotationQueueRequestBody | Body containing annotation queue update parameters. At least one field must be provided.

    try:
        # Update an annotation queue
        api_response = api_instance.annotation_queues_update(annotation_queue_id, update_annotation_queue_request_body)
        print("The response of AnnotationQueuesApi->annotation_queues_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationQueuesApi->annotation_queues_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_queue_id** | **str**| The unique identifier of the annotation queue | 
 **update_annotation_queue_request_body** | [**UpdateAnnotationQueueRequestBody**](UpdateAnnotationQueueRequestBody.md)| Body containing annotation queue update parameters. At least one field must be provided. | 

### Return type

[**AnnotationQueue**](AnnotationQueue.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Annotation queue successfully updated |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

