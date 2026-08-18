# arize._generated.api_client.TagsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_tag**](TagsApi.md#create_tag) | **POST** /v2/tags | Create a tag
[**delete_tag**](TagsApi.md#delete_tag) | **DELETE** /v2/tags/{tag_id} | Delete a tag
[**update_tag**](TagsApi.md#update_tag) | **PATCH** /v2/tags/{tag_id} | Update a tag


# **create_tag**
> Tag create_tag(create_tag_request)

Create a tag

Create a tag in a space. Tags are shared within the space and can then be
attached to resources across the platform.

**Payload Requirements**
- `name` is required, must be non-empty after trimming, and at most 100 characters.
- `name` must be unique within the space, compared **case-insensitively** — a
  space that already contains `Production` cannot also contain `production`.
  A collision returns `409`.
- `space_id` is required and must be a space the caller can create tags in.
- `color`, when provided, must be one of the `TagColor` values.
- System-managed fields (`id`, `created_at`, `updated_at`) are rejected on input.
- Unrecognized fields are rejected with `400` rather than ignored.

**Valid example**
```json
{
  "name": "production",
  "description": "Resources serving production traffic",
  "color": "GREEN",
  "space_id": "U3BhY2U6MTIzNDU="
}
```

**Invalid example** (name collides with an existing tag, differing only in case)

Request:
```json
{
  "name": "Production",
  "space_id": "U3BhY2U6MTIzNDU="
}
```

Response:
```json
{
  "type": "https://arize.com/docs/ax/rest-reference/errors#resource-conflict",
  "title": "Conflict",
  "status": 409,
  "detail": "A tag with this name already exists in the space",
  "request_id": "req_01HZY6X8E7"
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_tag_request import CreateTagRequest
from arize._generated.api_client.models.tag import Tag
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
    api_instance = arize._generated.api_client.TagsApi(api_client)
    create_tag_request = {"name":"production","description":"Resources serving production traffic","color":"GREEN","space_id":"U3BhY2U6MTIzNDU="} # CreateTagRequest | Body containing tag creation parameters

    try:
        # Create a tag
        api_response = api_instance.create_tag(create_tag_request)
        print("The response of TagsApi->create_tag:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TagsApi->create_tag: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_tag_request** | [**CreateTagRequest**](CreateTagRequest.md)| Body containing tag creation parameters | 

### Return type

[**Tag**](Tag.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | A tag object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_tag**
> delete_tag(tag_id)

Delete a tag

Delete a tag by its ID. This operation is irreversible.

The tag is detached from every resource it was attached to. The resources
themselves are not affected.

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
    api_instance = arize._generated.api_client.TagsApi(api_client)
    tag_id = 'VGFnOjEyMzQ1' # str | The unique tag identifier (base64)

    try:
        # Delete a tag
        api_instance.delete_tag(tag_id)
    except Exception as e:
        print("Exception when calling TagsApi->delete_tag: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **tag_id** | **str**| The unique tag identifier (base64) | 

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
**204** | Tag successfully deleted and detached from all resources |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_tag**
> Tag update_tag(tag_id, update_tag_request)

Update a tag

Update a tag's name, description, or color.

Tags are shared, so an update is visible on every resource the tag is
attached to.

**Payload Requirements**
- At least one of `name`, `description`, or `color` must be provided.
  Omitted fields are left unchanged.
- `name`, when provided, must be non-empty after trimming, at most 100
  characters, and unique within the space compared **case-insensitively**.
  A collision returns `409`.
- `description` and `color` accept `null` to clear the current value.
- System-managed fields (`id`, `created_at`, `updated_at`) cannot be modified.
  `updated_at` is advanced automatically.

**Valid example**
```json
{
  "name": "production-critical",
  "color": "RED"
}
```

**Invalid example** (empty body — nothing to update)

Request:
```json
{}
```

Response:
```json
{
  "type": "https://arize.com/docs/ax/rest-reference/errors#invalid-request",
  "title": "Bad Request",
  "status": 400,
  "detail": "At least one field (name, description, color) must be provided",
  "request_id": "req_01HZY6X8E7"
}
```

Unrecognized fields are rejected with `400` rather than ignored, so a
misspelled field name fails loudly instead of silently doing nothing.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.tag import Tag
from arize._generated.api_client.models.update_tag_request import UpdateTagRequest
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
    api_instance = arize._generated.api_client.TagsApi(api_client)
    tag_id = 'VGFnOjEyMzQ1' # str | The unique tag identifier (base64)
    update_tag_request = {"name":"production-critical","color":"RED"} # UpdateTagRequest | Body containing the tag fields to update. At least one of `name`, `description`, or `color` must be provided. 

    try:
        # Update a tag
        api_response = api_instance.update_tag(tag_id, update_tag_request)
        print("The response of TagsApi->update_tag:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling TagsApi->update_tag: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **tag_id** | **str**| The unique tag identifier (base64) | 
 **update_tag_request** | [**UpdateTagRequest**](UpdateTagRequest.md)| Body containing the tag fields to update. At least one of &#x60;name&#x60;, &#x60;description&#x60;, or &#x60;color&#x60; must be provided.  | 

### Return type

[**Tag**](Tag.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A tag object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

