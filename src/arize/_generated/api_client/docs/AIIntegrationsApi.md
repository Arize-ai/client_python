# arize._generated.api_client.AIIntegrationsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**ai_integrations_create**](AIIntegrationsApi.md#ai_integrations_create) | **POST** /v2/ai-integrations | Create an AI integration
[**ai_integrations_delete**](AIIntegrationsApi.md#ai_integrations_delete) | **DELETE** /v2/ai-integrations/{integration_id} | Delete an AI integration
[**ai_integrations_get**](AIIntegrationsApi.md#ai_integrations_get) | **GET** /v2/ai-integrations/{integration_id} | Get an AI integration
[**ai_integrations_list**](AIIntegrationsApi.md#ai_integrations_list) | **GET** /v2/ai-integrations | List AI integrations
[**ai_integrations_update**](AIIntegrationsApi.md#ai_integrations_update) | **PATCH** /v2/ai-integrations/{integration_id} | Update an AI integration


# **ai_integrations_create**
> AiIntegration ai_integrations_create(ai_integrations_create_request)

Create an AI integration

Create a new AI integration for an external LLM provider.

**Payload Requirements**
- `name` and `provider` are required.
- The integration name must be unique within the account.
- `provider` must be one of: `openAI`, `azureOpenAI`, `awsBedrock`, `vertexAI`, `anthropic`, `custom`.
- If `scopings` is omitted, the integration defaults to account-wide visibility.
- `enable_default_models` defaults to `false` if not provided.
- `function_calling_enabled` defaults to `true` if not provided.
- `auth_type` defaults to `default` if not provided.
- For `awsBedrock` provider, `provider_metadata` must include `role_arn`.
- For `vertexAI` provider, `provider_metadata` must include `project_id`, `location`, and `project_access_label`.

**Valid example**
```json
{
  "name": "Production OpenAI",
  "provider": "openAI",
  "api_key": "sk-abc123...",
  "model_names": ["gpt-4", "gpt-4o"],
  "enable_default_models": true
}
```

**Invalid example** (missing required `provider`)
```json
{
  "name": "My Integration"
}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.ai_integration import AiIntegration
from arize._generated.api_client.models.ai_integrations_create_request import AiIntegrationsCreateRequest
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
    api_instance = arize._generated.api_client.AIIntegrationsApi(api_client)
    ai_integrations_create_request = {"name":"Production OpenAI","provider":"openAI","api_key":"sk-abc123...","model_names":["gpt-4","gpt-4o"],"enable_default_models":true,"scopings":[{"organization_id":"QWNjb3VudE9yZzoxMjM6YWJj","space_id":null}]} # AiIntegrationsCreateRequest | Body containing AI integration creation parameters

    try:
        # Create an AI integration
        api_response = api_instance.ai_integrations_create(ai_integrations_create_request)
        print("The response of AIIntegrationsApi->ai_integrations_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AIIntegrationsApi->ai_integrations_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **ai_integrations_create_request** | [**AiIntegrationsCreateRequest**](AiIntegrationsCreateRequest.md)| Body containing AI integration creation parameters | 

### Return type

[**AiIntegration**](AiIntegration.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | An AI integration object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **ai_integrations_delete**
> ai_integrations_delete(integration_id)

Delete an AI integration

Delete an AI integration by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.AIIntegrationsApi(api_client)
    integration_id = 'TGxtSW50ZWdyYXRpb246MTI6YUJjRA==' # str | The unique identifier of the AI integration

    try:
        # Delete an AI integration
        api_instance.ai_integrations_delete(integration_id)
    except Exception as e:
        print("Exception when calling AIIntegrationsApi->ai_integrations_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **integration_id** | **str**| The unique identifier of the AI integration | 

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
**204** | AI integration successfully deleted |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **ai_integrations_get**
> AiIntegration ai_integrations_get(integration_id)

Get an AI integration

Get a specific AI integration by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.ai_integration import AiIntegration
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
    api_instance = arize._generated.api_client.AIIntegrationsApi(api_client)
    integration_id = 'TGxtSW50ZWdyYXRpb246MTI6YUJjRA==' # str | The unique identifier of the AI integration

    try:
        # Get an AI integration
        api_response = api_instance.ai_integrations_get(integration_id)
        print("The response of AIIntegrationsApi->ai_integrations_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AIIntegrationsApi->ai_integrations_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **integration_id** | **str**| The unique identifier of the AI integration | 

### Return type

[**AiIntegration**](AiIntegration.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An AI integration object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **ai_integrations_list**
> AiIntegrationsList200Response ai_integrations_list(space_id=space_id, name=name, limit=limit, cursor=cursor)

List AI integrations

List AI integrations the user has access to.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.ai_integrations_list200_response import AiIntegrationsList200Response
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
    api_instance = arize._generated.api_client.AIIntegrationsApi(api_client)
    space_id = 'space_id_example' # str | Filter search results to a particular space ID (optional)
    name = 'name_example' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. When omitted, no name filter is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List AI integrations
        api_response = api_instance.ai_integrations_list(space_id=space_id, name=name, limit=limit, cursor=cursor)
        print("The response of AIIntegrationsApi->ai_integrations_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AIIntegrationsApi->ai_integrations_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. When omitted, no name filter is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**AiIntegrationsList200Response**](AiIntegrationsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of AI integration objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **ai_integrations_update**
> AiIntegration ai_integrations_update(integration_id, ai_integrations_update_request)

Update an AI integration

Update an AI integration's configuration. At least one field must be provided.
Omitted fields are left unchanged.

**Payload Requirements**
- At least one of the updatable fields must be provided.
- `name`, if provided, must be unique within the account.
- `api_key` can be set to `null` to remove the existing key, or omitted to keep it unchanged.
- `scopings`, if provided, replaces all existing scoping rules.
- `provider_metadata`, if provided, replaces existing metadata. Set to `null` to remove.

**Valid example**
```json
{
  "name": "Updated Integration",
  "model_names": ["gpt-4o", "gpt-4o-mini"]
}
```

**Invalid example** (empty body)
```json
{}
```

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.ai_integration import AiIntegration
from arize._generated.api_client.models.ai_integrations_update_request import AiIntegrationsUpdateRequest
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
    api_instance = arize._generated.api_client.AIIntegrationsApi(api_client)
    integration_id = 'TGxtSW50ZWdyYXRpb246MTI6YUJjRA==' # str | The unique identifier of the AI integration
    ai_integrations_update_request = {"name":"Updated OpenAI Integration","api_key":null,"model_names":["gpt-4o","gpt-4o-mini"]} # AiIntegrationsUpdateRequest | Body containing AI integration update parameters. At least one field must be provided.

    try:
        # Update an AI integration
        api_response = api_instance.ai_integrations_update(integration_id, ai_integrations_update_request)
        print("The response of AIIntegrationsApi->ai_integrations_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AIIntegrationsApi->ai_integrations_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **integration_id** | **str**| The unique identifier of the AI integration | 
 **ai_integrations_update_request** | [**AiIntegrationsUpdateRequest**](AiIntegrationsUpdateRequest.md)| Body containing AI integration update parameters. At least one field must be provided. | 

### Return type

[**AiIntegration**](AiIntegration.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An AI integration object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

