# arize._generated.api_client.AnnotationConfigsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**annotation_configs_create**](AnnotationConfigsApi.md#annotation_configs_create) | **POST** /v2/annotation-configs | Create an annotation config
[**annotation_configs_delete**](AnnotationConfigsApi.md#annotation_configs_delete) | **DELETE** /v2/annotation-configs/{annotation_config_id} | Delete an annotation config
[**annotation_configs_get**](AnnotationConfigsApi.md#annotation_configs_get) | **GET** /v2/annotation-configs/{annotation_config_id} | Get an annotation config
[**annotation_configs_list**](AnnotationConfigsApi.md#annotation_configs_list) | **GET** /v2/annotation-configs | List annotation configs


# **annotation_configs_create**
> AnnotationConfig annotation_configs_create(create_annotation_config_request_body)

Create an annotation config

Create a new annotation config.

**Payload Requirements**
- The annotation config name must be unique within the given space.

**Valid example**
```json
{
  "name": "my-annotation-config",
  "space_id": "spc_123",
  "annotation_config_type": "categorical",
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
  "optimization_direction": "maximize"
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_config import AnnotationConfig
from arize._generated.api_client.models.create_annotation_config_request_body import CreateAnnotationConfigRequestBody
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
    create_annotation_config_request_body = {"name":"Accuracy","space_id":"space_12345","annotation_config_type":"categorical","values":[{"label":"accurate","score":1},{"label":"inaccurate","score":0}],"optimization_direction":"maximize"} # CreateAnnotationConfigRequestBody | Body containing annotation config creation parameters

    try:
        # Create an annotation config
        api_response = api_instance.annotation_configs_create(create_annotation_config_request_body)
        print("The response of AnnotationConfigsApi->annotation_configs_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->annotation_configs_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_annotation_config_request_body** | [**CreateAnnotationConfigRequestBody**](CreateAnnotationConfigRequestBody.md)| Body containing annotation config creation parameters | 

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
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_configs_delete**
> annotation_configs_delete(annotation_config_id)

Delete an annotation config

Delete an annotation config by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.AnnotationConfigsApi(api_client)
    annotation_config_id = 'annotation_config_id_example' # str | The unique identifier of the annotation config

    try:
        # Delete an annotation config
        api_instance.annotation_configs_delete(annotation_config_id)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->annotation_configs_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique identifier of the annotation config | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **annotation_configs_get**
> AnnotationConfig annotation_configs_get(annotation_config_id)

Get an annotation config

Get an annotation config object by its ID.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    annotation_config_id = 'annotation_config_id_example' # str | The unique identifier of the annotation config

    try:
        # Get an annotation config
        api_response = api_instance.annotation_configs_get(annotation_config_id)
        print("The response of AnnotationConfigsApi->annotation_configs_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->annotation_configs_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **annotation_config_id** | **str**| The unique identifier of the annotation config | 

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

# **annotation_configs_list**
> AnnotationConfigsList200Response annotation_configs_list(space_id=space_id, limit=limit, cursor=cursor)

List annotation configs

List annotation configs the user has access to.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.annotation_configs_list200_response import AnnotationConfigsList200Response
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
    space_id = 'space_id_example' # str | Filter search results to a particular space ID (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List annotation configs
        api_response = api_instance.annotation_configs_list(space_id=space_id, limit=limit, cursor=cursor)
        print("The response of AnnotationConfigsApi->annotation_configs_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AnnotationConfigsApi->annotation_configs_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**AnnotationConfigsList200Response**](AnnotationConfigsList200Response.md)

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
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

