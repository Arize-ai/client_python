# arize._generated.api_client.PromptsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_prompt**](PromptsApi.md#create_prompt) | **POST** /v2/prompts | Create a prompt
[**create_prompt_version**](PromptsApi.md#create_prompt_version) | **POST** /v2/prompts/{prompt_id}/versions | Create a prompt version
[**delete_prompt**](PromptsApi.md#delete_prompt) | **DELETE** /v2/prompts/{prompt_id} | Delete a prompt
[**delete_prompt_version_label**](PromptsApi.md#delete_prompt_version_label) | **DELETE** /v2/prompt-versions/{version_id}/labels/{label_name} | Remove a label from a prompt version
[**get_prompt**](PromptsApi.md#get_prompt) | **GET** /v2/prompts/{prompt_id} | Get a prompt
[**get_prompt_label**](PromptsApi.md#get_prompt_label) | **GET** /v2/prompts/{prompt_id}/labels/{label_name} | Resolve a label to a prompt version
[**get_prompt_version**](PromptsApi.md#get_prompt_version) | **GET** /v2/prompt-versions/{version_id} | Get a prompt version
[**list_prompt_versions**](PromptsApi.md#list_prompt_versions) | **GET** /v2/prompts/{prompt_id}/versions | List prompt versions
[**list_prompts**](PromptsApi.md#list_prompts) | **GET** /v2/prompts | List prompts
[**set_prompt_version_label**](PromptsApi.md#set_prompt_version_label) | **PUT** /v2/prompt-versions/{version_id}/labels | Set labels on a prompt version
[**update_prompt**](PromptsApi.md#update_prompt) | **PATCH** /v2/prompts/{prompt_id} | Update a prompt


# **create_prompt**
> PromptWithVersion create_prompt(create_prompt_request)

Create a prompt

Create a new prompt with an initial version.

**Payload Requirements**
- The prompt name must be unique within the given space.
- At least one message is required.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_prompt_request import CreatePromptRequest
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
    create_prompt_request = {"space_id":"U3BhY2U6MTIzOmFiY2Q=","name":"My Prompt","description":"A helpful assistant prompt","version":{"commit_message":"Initial version","input_variable_format":"F_STRING","provider":"OPEN_AI","model":"gpt-4","messages":[{"role":"SYSTEM","content":"You are a helpful assistant."},{"role":"USER","content":"Hello, {name}!"}]}} # CreatePromptRequest | Body containing prompt creation parameters with an initial version

    try:
        # Create a prompt
        api_response = api_instance.create_prompt(create_prompt_request)
        print("The response of PromptsApi->create_prompt:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->create_prompt: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_prompt_request** | [**CreatePromptRequest**](CreatePromptRequest.md)| Body containing prompt creation parameters with an initial version | 

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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_prompt_version**
> PromptVersion create_prompt_version(prompt_id, create_prompt_version_request)

Create a prompt version

Create a new version of an existing prompt.

**Payload Requirements**
- A `commit_message` is required.
- At least one message is required in `messages`.
- Do not include system-managed fields on input: `id`, `commit_hash`, `created_at`, `created_by_user_id`.
  Requests that contain these fields will be rejected.
- `provider` is required. `input_variable_format` defaults to `F_STRING` if not provided.

**Valid example** (create)
```json
{
  "commit_message": "Updated system prompt for better responses",
  "input_variable_format": "F_STRING",
  "provider": "OPEN_AI",
  "model": "gpt-4",
  "messages": [
    {
      "role": "SYSTEM",
      "content": "You are a helpful assistant."
    },
    {
      "role": "USER",
      "content": "Hello, {name}!"
    }
  ]
}
```

**Invalid example** (missing required `commit_message`)
```json
{
  "input_variable_format": "F_STRING",
  "provider": "OPEN_AI",
  "messages": [
    {
      "role": "USER",
      "content": "Hello!"
    }
  ]
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_prompt_version_request import CreatePromptVersionRequest
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
    prompt_id = 'UHJvbXB0OjEyMzQ1' # str | The unique prompt identifier (base64)
    create_prompt_version_request = {"commit_message":"Updated system prompt for better responses","input_variable_format":"F_STRING","provider":"OPEN_AI","model":"gpt-4","messages":[{"role":"SYSTEM","content":"You are a helpful assistant."},{"role":"USER","content":"Hello, {name}!"}]} # CreatePromptVersionRequest | Body containing prompt version creation parameters

    try:
        # Create a prompt version
        api_response = api_instance.create_prompt_version(prompt_id, create_prompt_version_request)
        print("The response of PromptsApi->create_prompt_version:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->create_prompt_version: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique prompt identifier (base64) | 
 **create_prompt_version_request** | [**CreatePromptVersionRequest**](CreatePromptVersionRequest.md)| Body containing prompt version creation parameters | 

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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_prompt**
> delete_prompt(prompt_id)

Delete a prompt

Delete a prompt by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    prompt_id = 'UHJvbXB0OjEyMzQ1' # str | The unique prompt identifier (base64)

    try:
        # Delete a prompt
        api_instance.delete_prompt(prompt_id)
    except Exception as e:
        print("Exception when calling PromptsApi->delete_prompt: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique prompt identifier (base64) | 

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

# **delete_prompt_version_label**
> delete_prompt_version_label(version_id, label_name)

Remove a label from a prompt version

Remove a specific label from a prompt version.

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
    api_instance = arize._generated.api_client.PromptsApi(api_client)
    version_id = 'UHJvbXB0VmVyc2lvbjoxMjM0NQ==' # str | The unique prompt version identifier (base64)
    label_name = 'label_name_example' # str | The name of the label (e.g., \"production\", \"staging\")

    try:
        # Remove a label from a prompt version
        api_instance.delete_prompt_version_label(version_id, label_name)
    except Exception as e:
        print("Exception when calling PromptsApi->delete_prompt_version_label: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique prompt version identifier (base64) | 
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

# **get_prompt**
> PromptWithVersion get_prompt(prompt_id, version_id=version_id, label=label)

Get a prompt

Get a specific prompt by its ID. The response always includes a resolved
version. By default, the latest version is returned. Use the `version_id`
or `label` query parameter to resolve a specific version instead. You
cannot supply both `version_id` and `label`.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    prompt_id = 'UHJvbXB0OjEyMzQ1' # str | The unique prompt identifier (base64)
    version_id = 'UHJvbXB0VmVyc2lvbjoxMjM0NQ==' # str | Return the prompt with this specific version (base64 identifier (base64)). Mutually exclusive with `label`. (optional)
    label = 'production' # str | Return the prompt with the version pointed to by this label (e.g., \"production\"). Mutually exclusive with `version_id`. (optional)

    try:
        # Get a prompt
        api_response = api_instance.get_prompt(prompt_id, version_id=version_id, label=label)
        print("The response of PromptsApi->get_prompt:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->get_prompt: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique prompt identifier (base64) | 
 **version_id** | **str**| Return the prompt with this specific version (base64 identifier (base64)). Mutually exclusive with &#x60;label&#x60;. | [optional] 
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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_prompt_label**
> PromptVersion get_prompt_label(prompt_id, label_name)

Resolve a label to a prompt version

Resolve a label on a prompt to the version it points to. Returns the
full `PromptVersion` object that this label currently references.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    prompt_id = 'UHJvbXB0OjEyMzQ1' # str | The unique prompt identifier (base64)
    label_name = 'label_name_example' # str | The name of the label (e.g., \"production\", \"staging\")

    try:
        # Resolve a label to a prompt version
        api_response = api_instance.get_prompt_label(prompt_id, label_name)
        print("The response of PromptsApi->get_prompt_label:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->get_prompt_label: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique prompt identifier (base64) | 
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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_prompt_version**
> PromptVersion get_prompt_version(version_id)

Get a prompt version

Get a specific prompt version by its ID.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    version_id = 'UHJvbXB0VmVyc2lvbjoxMjM0NQ==' # str | The unique prompt version identifier (base64)

    try:
        # Get a prompt version
        api_response = api_instance.get_prompt_version(version_id)
        print("The response of PromptsApi->get_prompt_version:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->get_prompt_version: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique prompt version identifier (base64) | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_prompt_versions**
> ListPromptVersionsResponse list_prompt_versions(prompt_id, limit=limit, cursor=cursor)

List prompt versions

List all versions of a prompt, sorted by creation date with the most
recently created versions first.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_prompt_versions_response import ListPromptVersionsResponse
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
    prompt_id = 'UHJvbXB0OjEyMzQ1' # str | The unique prompt identifier (base64)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List prompt versions
        api_response = api_instance.list_prompt_versions(prompt_id, limit=limit, cursor=cursor)
        print("The response of PromptsApi->list_prompt_versions:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->list_prompt_versions: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique prompt identifier (base64) | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListPromptVersionsResponse**](ListPromptVersionsResponse.md)

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

# **list_prompts**
> ListPromptsResponse list_prompts(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List prompts

List prompts the user has access to.

The prompts are sorted by update date, with the most recently updated
prompts coming first.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_prompts_response import ListPromptsResponse
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
        api_response = api_instance.list_prompts(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of PromptsApi->list_prompts:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->list_prompts: %s\n" % e)
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

[**ListPromptsResponse**](ListPromptsResponse.md)

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **set_prompt_version_label**
> PromptVersion set_prompt_version_label(version_id, set_prompt_version_labels_request)

Set labels on a prompt version

Set (replace) all labels on a prompt version. This is an idempotent
operation. If a label already exists on another version of the same
prompt, it will be moved to this version.

Labels not included in the request will be removed from this version.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt_version import PromptVersion
from arize._generated.api_client.models.set_prompt_version_labels_request import SetPromptVersionLabelsRequest
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
    version_id = 'UHJvbXB0VmVyc2lvbjoxMjM0NQ==' # str | The unique prompt version identifier (base64)
    set_prompt_version_labels_request = {"labels":["production","staging"]} # SetPromptVersionLabelsRequest | Body containing the labels to set on a prompt version

    try:
        # Set labels on a prompt version
        api_response = api_instance.set_prompt_version_label(version_id, set_prompt_version_labels_request)
        print("The response of PromptsApi->set_prompt_version_label:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->set_prompt_version_label: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique prompt version identifier (base64) | 
 **set_prompt_version_labels_request** | [**SetPromptVersionLabelsRequest**](SetPromptVersionLabelsRequest.md)| Body containing the labels to set on a prompt version | 

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
**200** | A prompt version object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_prompt**
> Prompt update_prompt(prompt_id, update_prompt_request)

Update a prompt

Update a prompt's metadata by its ID. Currently supports updating the
description. The prompt name is immutable after creation; to rename a
prompt, delete it and create a new one (note: this loses version history).

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.prompt import Prompt
from arize._generated.api_client.models.update_prompt_request import UpdatePromptRequest
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
    prompt_id = 'UHJvbXB0OjEyMzQ1' # str | The unique prompt identifier (base64)
    update_prompt_request = {"description":"Updated prompt description"} # UpdatePromptRequest | Body containing prompt update parameters. At least one field must be provided.

    try:
        # Update a prompt
        api_response = api_instance.update_prompt(prompt_id, update_prompt_request)
        print("The response of PromptsApi->update_prompt:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling PromptsApi->update_prompt: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **prompt_id** | **str**| The unique prompt identifier (base64) | 
 **update_prompt_request** | [**UpdatePromptRequest**](UpdatePromptRequest.md)| Body containing prompt update parameters. At least one field must be provided. | 

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
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

