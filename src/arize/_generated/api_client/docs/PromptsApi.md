# arize._generated.api_client.PromptsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**prompt_labels_get**](PromptsApi.md#prompt_labels_get) | **GET** /v2/prompts/{prompt_id}/labels/{label_name} | Resolve a label to a prompt version
[**prompt_version_labels_delete**](PromptsApi.md#prompt_version_labels_delete) | **DELETE** /v2/prompt-versions/{version_id}/labels/{label_name} | Remove a label from a prompt version
[**prompt_version_labels_set**](PromptsApi.md#prompt_version_labels_set) | **PUT** /v2/prompt-versions/{version_id}/labels | Set labels on a prompt version
[**prompt_versions_create**](PromptsApi.md#prompt_versions_create) | **POST** /v2/prompts/{prompt_id}/versions | Create a prompt version
[**prompt_versions_get**](PromptsApi.md#prompt_versions_get) | **GET** /v2/prompt-versions/{version_id} | Get a prompt version
[**prompt_versions_list**](PromptsApi.md#prompt_versions_list) | **GET** /v2/prompts/{prompt_id}/versions | List prompt versions
[**prompts_create**](PromptsApi.md#prompts_create) | **POST** /v2/prompts | Create a prompt
[**prompts_delete**](PromptsApi.md#prompts_delete) | **DELETE** /v2/prompts/{prompt_id} | Delete a prompt
[**prompts_get**](PromptsApi.md#prompts_get) | **GET** /v2/prompts/{prompt_id} | Get a prompt
[**prompts_list**](PromptsApi.md#prompts_list) | **GET** /v2/prompts | List prompts
[**prompts_update**](PromptsApi.md#prompts_update) | **PATCH** /v2/prompts/{prompt_id} | Update a prompt


# **prompt_labels_get**
> PromptVersion prompt_labels_get(prompt_id, label_name)

Resolve a label to a prompt version

Resolve a label on a prompt to the version it points to. Returns the
full `PromptVersion` object that this label currently references.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_version import PromptVersion
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'prompt_12345' # str | The unique identifier of the prompt
    label_name = 'label_name_example' # str | The name of the label (e.g., \"production\", \"staging\")

    try:
        # Resolve a label to a prompt version
        api_response = api_instance.prompt_labels_get(prompt_id, label_name)
        print("The response of PromptsApi->prompt_labels_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompt_labels_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique identifier of the prompt | 
 **label_name** | **str**| The name of the label (e.g., \&quot;production\&quot;, \&quot;staging\&quot;) | 

### Return type

[**PromptVersion**](PromptVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A prompt version object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompt_version_labels_delete**
> prompt_version_labels_delete(version_id, label_name)

Remove a label from a prompt version

Remove a specific label from a prompt version.

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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    version_id = 'pv_12345' # str | The unique identifier of the prompt version
    label_name = 'label_name_example' # str | The name of the label (e.g., \"production\", \"staging\")

    try:
        # Remove a label from a prompt version
        api_instance.prompt_version_labels_delete(version_id, label_name)
    except Exception as e:
        print("Exception when calling PromptsApi->prompt_version_labels_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique identifier of the prompt version | 
 **label_name** | **str**| The name of the label (e.g., \&quot;production\&quot;, \&quot;staging\&quot;) | 

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
**204** | Label successfully removed from version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompt_version_labels_set**
> PromptVersionLabelsSet200Response prompt_version_labels_set(version_id, prompt_version_labels_set_request)

Set labels on a prompt version

Set (replace) all labels on a prompt version. This is an idempotent
operation. If a label already exists on another version of the same
prompt, it will be moved to this version.

Labels not included in the request will be removed from this version.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_version_labels_set200_response import PromptVersionLabelsSet200Response
from arize._generated.api_client.models.prompt_version_labels_set_request import PromptVersionLabelsSetRequest
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    version_id = 'pv_12345' # str | The unique identifier of the prompt version
    prompt_version_labels_set_request = {"labels":["production","staging"]} # PromptVersionLabelsSetRequest | Body containing the labels to set on a prompt version

    try:
        # Set labels on a prompt version
        api_response = api_instance.prompt_version_labels_set(version_id, prompt_version_labels_set_request)
        print("The response of PromptsApi->prompt_version_labels_set:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompt_version_labels_set: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique identifier of the prompt version | 
 **prompt_version_labels_set_request** | [**PromptVersionLabelsSetRequest**](PromptVersionLabelsSetRequest.md)| Body containing the labels to set on a prompt version | 

### Return type

[**PromptVersionLabelsSet200Response**](PromptVersionLabelsSet200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the labels set on a prompt version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompt_versions_create**
> PromptVersion prompt_versions_create(prompt_id, prompt_versions_create_request)

Create a prompt version

Create a new version of an existing prompt.

**Payload Requirements**
- A `commit_message` is required.
- At least one message is required in `messages`.
- Do not include system-managed fields on input: `id`, `commit_hash`, `created_at`, `created_by_user_id`.
  Requests that contain these fields will be rejected.
- `provider` is required. `input_variable_format` defaults to `f_string` if not provided.

**Valid example** (create)
```json
{
  "commit_message": "Updated system prompt for better responses",
  "input_variable_format": "f_string",
  "provider": "open_ai",
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, {name}!"
    }
  ]
}
```

**Invalid example** (missing required `commit_message`)
```json
{
  "input_variable_format": "f_string",
  "provider": "open_ai",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_version import PromptVersion
from arize._generated.api_client.models.prompt_versions_create_request import PromptVersionsCreateRequest
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'prompt_12345' # str | The unique identifier of the prompt
    prompt_versions_create_request = {"commit_message":"Updated system prompt for better responses","input_variable_format":"f_string","provider":"open_ai","model":"gpt-4","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello, {name}!"}]} # PromptVersionsCreateRequest | Body containing prompt version creation parameters

    try:
        # Create a prompt version
        api_response = api_instance.prompt_versions_create(prompt_id, prompt_versions_create_request)
        print("The response of PromptsApi->prompt_versions_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompt_versions_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique identifier of the prompt | 
 **prompt_versions_create_request** | [**PromptVersionsCreateRequest**](PromptVersionsCreateRequest.md)| Body containing prompt version creation parameters | 

### Return type

[**PromptVersion**](PromptVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | A prompt version object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompt_versions_get**
> PromptVersion prompt_versions_get(version_id)

Get a prompt version

Get a specific prompt version by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_version import PromptVersion
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    version_id = 'pv_12345' # str | The unique identifier of the prompt version

    try:
        # Get a prompt version
        api_response = api_instance.prompt_versions_get(version_id)
        print("The response of PromptsApi->prompt_versions_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompt_versions_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique identifier of the prompt version | 

### Return type

[**PromptVersion**](PromptVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A prompt version object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompt_versions_list**
> PromptVersionsList200Response prompt_versions_list(prompt_id, limit=limit, cursor=cursor)

List prompt versions

List all versions of a prompt, sorted by creation date with the most
recently created versions first.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_versions_list200_response import PromptVersionsList200Response
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'prompt_12345' # str | The unique identifier of the prompt
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List prompt versions
        api_response = api_instance.prompt_versions_list(prompt_id, limit=limit, cursor=cursor)
        print("The response of PromptsApi->prompt_versions_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompt_versions_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique identifier of the prompt | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**PromptVersionsList200Response**](PromptVersionsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of prompt version objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompts_create**
> PromptWithVersion prompts_create(prompts_create_request)

Create a prompt

Create a new prompt with an initial version.

**Payload Requirements**
- The prompt name must be unique within the given space.
- At least one message is required.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_with_version import PromptWithVersion
from arize._generated.api_client.models.prompts_create_request import PromptsCreateRequest
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompts_create_request = {"space_id":"U3BhY2U6MTIzOmFiY2Q=","name":"My Prompt","description":"A helpful assistant prompt","version":{"commit_message":"Initial version","input_variable_format":"f_string","provider":"open_ai","model":"gpt-4","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello, {name}!"}]}} # PromptsCreateRequest | Body containing prompt creation parameters with an initial version

    try:
        # Create a prompt
        api_response = api_instance.prompts_create(prompts_create_request)
        print("The response of PromptsApi->prompts_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompts_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompts_create_request** | [**PromptsCreateRequest**](PromptsCreateRequest.md)| Body containing prompt creation parameters with an initial version | 

### Return type

[**PromptWithVersion**](PromptWithVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | A prompt object with a resolved version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompts_delete**
> prompts_delete(prompt_id)

Delete a prompt

Delete a prompt by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'prompt_12345' # str | The unique identifier of the prompt

    try:
        # Delete a prompt
        api_instance.prompts_delete(prompt_id)
    except Exception as e:
        print("Exception when calling PromptsApi->prompts_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique identifier of the prompt | 

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
**204** | Prompt successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompts_get**
> PromptWithVersion prompts_get(prompt_id, version_id=version_id, label=label)

Get a prompt

Get a specific prompt by its ID. The response always includes a resolved
version. By default, the latest version is returned. Use the `version_id`
or `label` query parameter to resolve a specific version instead. You
cannot supply both `version_id` and `label`.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_with_version import PromptWithVersion
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'prompt_12345' # str | The unique identifier of the prompt
    version_id = 'pv_12345' # str | Return the prompt with this specific version. Mutually exclusive with `label`. (optional)
    label = 'production' # str | Return the prompt with the version pointed to by this label (e.g., \"production\"). Mutually exclusive with `version_id`. (optional)

    try:
        # Get a prompt
        api_response = api_instance.prompts_get(prompt_id, version_id=version_id, label=label)
        print("The response of PromptsApi->prompts_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompts_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique identifier of the prompt | 
 **version_id** | **str**| Return the prompt with this specific version. Mutually exclusive with &#x60;label&#x60;. | [optional] 
 **label** | **str**| Return the prompt with the version pointed to by this label (e.g., \&quot;production\&quot;). Mutually exclusive with &#x60;version_id&#x60;. | [optional] 

### Return type

[**PromptWithVersion**](PromptWithVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A prompt object with a resolved version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompts_list**
> PromptsList200Response prompts_list(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List prompts

List prompts the user has access to.

The prompts are sorted by update date, with the most recently updated
prompts coming first.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompts_list200_response import PromptsList200Response
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List prompts
        api_response = api_instance.prompts_list(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of PromptsApi->prompts_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompts_list: %s\n" % e)
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

[**PromptsList200Response**](PromptsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of prompt objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **prompts_update**
> Prompt prompts_update(prompt_id, prompts_update_request)

Update a prompt

Update a prompt's metadata by its ID. Currently supports updating the
description. The prompt name is immutable after creation; to rename a
prompt, delete it and create a new one (note: this loses version history).

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt import Prompt
from arize._generated.api_client.models.prompts_update_request import PromptsUpdateRequest
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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'prompt_12345' # str | The unique identifier of the prompt
    prompts_update_request = {"description":"Updated prompt description"} # PromptsUpdateRequest | Body containing prompt update parameters. At least one field must be provided.

    try:
        # Update a prompt
        api_response = api_instance.prompts_update(prompt_id, prompts_update_request)
        print("The response of PromptsApi->prompts_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->prompts_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique identifier of the prompt | 
 **prompts_update_request** | [**PromptsUpdateRequest**](PromptsUpdateRequest.md)| Body containing prompt update parameters. At least one field must be provided. | 

### Return type

[**Prompt**](Prompt.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A prompt object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

