# arize._generated.api_client.AnnotationConfigsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**add_annotation_config_tags**](AnnotationConfigsApi.md#add_annotation_config_tags) | **POST** /v2/annotation-configs/{annotation_config_id}/tags | Attach tags to a annotation config
[**create_annotation_config**](AnnotationConfigsApi.md#create_annotation_config) | **POST** /v2/annotation-configs | Create an annotation config
[**delete_annotation_config**](AnnotationConfigsApi.md#delete_annotation_config) | **DELETE** /v2/annotation-configs/{annotation_config_id} | Delete an annotation config
[**get_annotation_config**](AnnotationConfigsApi.md#get_annotation_config) | **GET** /v2/annotation-configs/{annotation_config_id} | Get an annotation config
[**list_annotation_config_tags**](AnnotationConfigsApi.md#list_annotation_config_tags) | **GET** /v2/annotation-configs/{annotation_config_id}/tags | List tags on an annotation config
[**list_annotation_configs**](AnnotationConfigsApi.md#list_annotation_configs) | **GET** /v2/annotation-configs | List annotation configs
[**update_annotation_config**](AnnotationConfigsApi.md#update_annotation_config) | **PATCH** /v2/annotation-configs/{annotation_config_id} | Update an annotation config


# **add_annotation_config_tags**
> ListTagsResponse add_annotation_config_tags(annotation_config_id, add_tags_request)

Attach tags to a annotation config

Attach one or more existing tags to a annotation config.

**Payload Requirements**
- `tag_ids` is required and must contain between 1 and 100 tag IDs.
- Every tag must already exist and belong to the same space as the
  annotation config. A tag from another space returns `422`.
- Attaching a tag that is already attached is idempotent, so the same
  request can be retried safely.
- Unrecognized fields are rejected with `422` rather than ignored.

Returns `200` with the annotation config's complete tag list, not `201`: attaching
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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    annotation_config_id = 'QW5ub3RhdGlvbkNvbmZpZzoxMjM0NQ==' # str | The unique annotation config identifier (base64)
    add_tags_request = {"tag_ids":["VGFnOjEyMzQ1","VGFnOjEyMzQ2"]} # AddTagsRequest | Body containing the IDs of the tags to attach to the resource

    try:
        # Attach tags to a annotation config
        api_response = api_instance.add_annotation_config_tags(annotation_config_id, add_tags_request)
        print("The response of AnnotationConfigsApi->add_annotation_config_tags:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->add_annotation_config_tags: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique annotation config identifier (base64) | 
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

# **create_annotation_config**
> AnnotationConfig create_annotation_config(create_annotation_config_request)

Create an annotation config

Create a new annotation config.

**Payload Requirements**
- The annotation config name must be unique within the given space.

**Valid example**
```json
{
  "name": "my-annotation-config",
  "space_id": "spc_123",
  "annotation_config_type": "CATEGORICAL",
  "values": [
    {
      "label": "value1",
      "score": 0
    },
    {
      "label": "value2",
      "score": 1
    }
  ],
  "optimization_direction": "MAXIMIZE"
}
```


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_config import AnnotationConfig
from arize._generated.api_client.models.create_annotation_config_request import CreateAnnotationConfigRequest
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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    create_annotation_config_request = {"name":"Accuracy","space_id":"space_12345","annotation_config_type":"CATEGORICAL","values":[{"label":"accurate","score":1},{"label":"inaccurate","score":0}],"optimization_direction":"MAXIMIZE"} # CreateAnnotationConfigRequest | Body containing annotation config creation parameters

    try:
        # Create an annotation config
        api_response = api_instance.create_annotation_config(create_annotation_config_request)
        print("The response of AnnotationConfigsApi->create_annotation_config:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->create_annotation_config: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_annotation_config_request** | [**CreateAnnotationConfigRequest**](CreateAnnotationConfigRequest.md)| Body containing annotation config creation parameters | 

### Return type

[**AnnotationConfig**](AnnotationConfig.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | An annotation config object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_annotation_config**
> delete_annotation_config(annotation_config_id)

Delete an annotation config

Delete an annotation config by its ID. The annotation config must not be associated
with an active annotation queue; remove it from those queues before deleting it.
This operation is irreversible.


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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    annotation_config_id = 'QW5ub3RhdGlvbkNvbmZpZzoxMjM0NQ==' # str | The unique annotation config identifier (base64)

    try:
        # Delete an annotation config
        api_instance.delete_annotation_config(annotation_config_id)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->delete_annotation_config: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique annotation config identifier (base64) | 

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
**204** | Annotation config successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_annotation_config**
> AnnotationConfig get_annotation_config(annotation_config_id)

Get an annotation config

Get an annotation config object by its ID.


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_config import AnnotationConfig
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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    annotation_config_id = 'QW5ub3RhdGlvbkNvbmZpZzoxMjM0NQ==' # str | The unique annotation config identifier (base64)

    try:
        # Get an annotation config
        api_response = api_instance.get_annotation_config(annotation_config_id)
        print("The response of AnnotationConfigsApi->get_annotation_config:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->get_annotation_config: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique annotation config identifier (base64) | 

### Return type

[**AnnotationConfig**](AnnotationConfig.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An annotation config object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_annotation_config_tags**
> ListTagsResponse list_annotation_config_tags(annotation_config_id)

List tags on an annotation config

List the tags attached to an annotation config.

Tags are shared within the space, so the same tag may appear on many
resources. An annotation config with no tags returns an empty list rather than a
404.

Requires read access to the annotation config. A caller who cannot read it receives
`404`, identical to the response for an annotation config that does not exist.

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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    annotation_config_id = 'QW5ub3RhdGlvbkNvbmZpZzoxMjM0NQ==' # str | The unique annotation config identifier (base64)

    try:
        # List tags on an annotation config
        api_response = api_instance.list_annotation_config_tags(annotation_config_id)
        print("The response of AnnotationConfigsApi->list_annotation_config_tags:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->list_annotation_config_tags: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique annotation config identifier (base64) | 

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

# **list_annotation_configs**
> ListAnnotationConfigsResponse list_annotation_configs(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List annotation configs

List annotation configs the user has access to.


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_annotation_configs_response import ListAnnotationConfigsResponse
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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return. Defaults to 50 if omitted; maximum is 100. (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List annotation configs
        api_response = api_instance.list_annotation_configs(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of AnnotationConfigsApi->list_annotation_configs:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->list_annotation_configs: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return. Defaults to 50 if omitted; maximum is 100. | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListAnnotationConfigsResponse**](ListAnnotationConfigsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of annotation config objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_annotation_config**
> AnnotationConfig update_annotation_config(annotation_config_id, update_annotation_config_request)

Update an annotation config

Update an annotation config by its ID.

**Payload Requirements**
- `annotation_config_type` is required and must match the stored config's type. The
  type is immutable and cannot be changed.
- The updatable fields depend on the type:
  - `CATEGORICAL`: `name`, `values`, `optimization_direction`.
  - `CONTINUOUS`: `name`, `minimum_score`, `maximum_score`, `optimization_direction`.
  - `FREEFORM`: `name`.
- All fields other than `annotation_config_type` are optional; omitted fields are left
  unchanged.
- `name`, if provided, must be unique within the space (409 Conflict if duplicate).
- `values` replaces the full label set (2-100 labels).
- System-managed fields (`id`, `space_id`, `created_at`) cannot be modified.

**Valid example** (categorical config)
```json
{
  "annotation_config_type": "CATEGORICAL",
  "name": "quality-v2",
  "values": [
    { "label": "good", "score": 1 },
    { "label": "bad", "score": 0 }
  ],
  "optimization_direction": "MAXIMIZE"
}
```

**Invalid example** (missing `annotation_config_type`)
```json
{
  "name": "quality-v2"
}
```


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_config import AnnotationConfig
from arize._generated.api_client.models.update_annotation_config_request import UpdateAnnotationConfigRequest
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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    annotation_config_id = 'QW5ub3RhdGlvbkNvbmZpZzoxMjM0NQ==' # str | The unique annotation config identifier (base64)
    update_annotation_config_request = {"annotation_config_type":"CATEGORICAL","name":"quality-v2","values":[{"label":"good","score":1},{"label":"bad","score":0}],"optimization_direction":"MAXIMIZE"} # UpdateAnnotationConfigRequest | Body containing annotation config update parameters. The annotation_config_type is required and must match the stored config's type.

    try:
        # Update an annotation config
        api_response = api_instance.update_annotation_config(annotation_config_id, update_annotation_config_request)
        print("The response of AnnotationConfigsApi->update_annotation_config:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->update_annotation_config: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique annotation config identifier (base64) | 
 **update_annotation_config_request** | [**UpdateAnnotationConfigRequest**](UpdateAnnotationConfigRequest.md)| Body containing annotation config update parameters. The annotation_config_type is required and must match the stored config&#39;s type. | 

### Return type

[**AnnotationConfig**](AnnotationConfig.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An annotation config object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

