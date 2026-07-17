# arize._generated.api_client.IntegrationsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_integration**](IntegrationsApi.md#create_integration) | **POST** /v2/integrations | Create an integration
[**delete_integration**](IntegrationsApi.md#delete_integration) | **DELETE** /v2/integrations/{integration_id} | Delete an integration
[**get_integration**](IntegrationsApi.md#get_integration) | **GET** /v2/integrations/{integration_id} | Get an integration
[**list_integrations**](IntegrationsApi.md#list_integrations) | **GET** /v2/integrations | List integrations
[**update_integration**](IntegrationsApi.md#update_integration) | **PATCH** /v2/integrations/{integration_id} | Update an integration


# **create_integration**
> LlmIntegration create_integration(body)

Create an integration

Create a new integration. The `type` field selects the config shape.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_llm_integration_request import CreateLlmIntegrationRequest
from arize._generated.api_client.models.llm_integration import LlmIntegration
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
    api_instance = arize._generated.api_client.IntegrationsApi(api_client)
    body = arize._generated.api_client.CreateLlmIntegrationRequest() # CreateLlmIntegrationRequest | Create a new integration. The `type` field selects the config shape; for `LLM`, `config.provider` selects the per-provider config.  **Payload Requirements** - `type`, `name`, and `config` are required. - `name` must be unique within the account for the given `type`. - For `type: LLM`, `config.provider` is required. Each provider's config   defines its own required fields — see the per-provider `config` schema. - `config.is_function_calling_enabled` defaults to `true` when omitted. - `scopings` defaults to account-wide visibility when omitted.  **Valid example** ```json {   \"type\": \"LLM\",   \"name\": \"Production OpenAI\",   \"config\": {     \"provider\": \"OPEN_AI\",     \"api_key\": \"sk-abc123...\"   } } ```  **Invalid example** (missing required `config`) ```json {   \"type\": \"LLM\",   \"name\": \"Bad Integration\" } ```  **Invalid example** (missing required `config.provider` for `type: LLM`) ```json {   \"type\": \"LLM\",   \"name\": \"Bad Integration\",   \"config\": {} } ```  **Invalid example** (missing required `config.api_key` for `OPEN_AI`) ```json {   \"type\": \"LLM\",   \"name\": \"Bad OpenAI\",   \"config\": { \"provider\": \"OPEN_AI\" } } ``` 

    try:
        # Create an integration
        api_response = api_instance.create_integration(body)
        print("The response of IntegrationsApi->create_integration:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling IntegrationsApi->create_integration: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body** | **CreateLlmIntegrationRequest**| Create a new integration. The &#x60;type&#x60; field selects the config shape; for &#x60;LLM&#x60;, &#x60;config.provider&#x60; selects the per-provider config.  **Payload Requirements** - &#x60;type&#x60;, &#x60;name&#x60;, and &#x60;config&#x60; are required. - &#x60;name&#x60; must be unique within the account for the given &#x60;type&#x60;. - For &#x60;type: LLM&#x60;, &#x60;config.provider&#x60; is required. Each provider&#39;s config   defines its own required fields — see the per-provider &#x60;config&#x60; schema. - &#x60;config.is_function_calling_enabled&#x60; defaults to &#x60;true&#x60; when omitted. - &#x60;scopings&#x60; defaults to account-wide visibility when omitted.  **Valid example** &#x60;&#x60;&#x60;json {   \&quot;type\&quot;: \&quot;LLM\&quot;,   \&quot;name\&quot;: \&quot;Production OpenAI\&quot;,   \&quot;config\&quot;: {     \&quot;provider\&quot;: \&quot;OPEN_AI\&quot;,     \&quot;api_key\&quot;: \&quot;sk-abc123...\&quot;   } } &#x60;&#x60;&#x60;  **Invalid example** (missing required &#x60;config&#x60;) &#x60;&#x60;&#x60;json {   \&quot;type\&quot;: \&quot;LLM\&quot;,   \&quot;name\&quot;: \&quot;Bad Integration\&quot; } &#x60;&#x60;&#x60;  **Invalid example** (missing required &#x60;config.provider&#x60; for &#x60;type: LLM&#x60;) &#x60;&#x60;&#x60;json {   \&quot;type\&quot;: \&quot;LLM\&quot;,   \&quot;name\&quot;: \&quot;Bad Integration\&quot;,   \&quot;config\&quot;: {} } &#x60;&#x60;&#x60;  **Invalid example** (missing required &#x60;config.api_key&#x60; for &#x60;OPEN_AI&#x60;) &#x60;&#x60;&#x60;json {   \&quot;type\&quot;: \&quot;LLM\&quot;,   \&quot;name\&quot;: \&quot;Bad OpenAI\&quot;,   \&quot;config\&quot;: { \&quot;provider\&quot;: \&quot;OPEN_AI\&quot; } } &#x60;&#x60;&#x60;  | 

### Return type

[**LlmIntegration**](LlmIntegration.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | An integration object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_integration**
> delete_integration(integration_id)

Delete an integration

Delete an integration by its ID. This operation is irreversible.

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
    api_instance = arize._generated.api_client.IntegrationsApi(api_client)
    integration_id = 'TGxtSW50ZWdyYXRpb246MTI6YUJjRA==' # str | The unique integration identifier (base64 global ID).

    try:
        # Delete an integration
        api_instance.delete_integration(integration_id)
    except Exception as e:
        print("Exception when calling IntegrationsApi->delete_integration: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **integration_id** | **str**| The unique integration identifier (base64 global ID). | 

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
**204** | Integration successfully deleted. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_integration**
> LlmIntegration get_integration(integration_id)

Get an integration

Get a specific integration by its ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.llm_integration import LlmIntegration
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
    api_instance = arize._generated.api_client.IntegrationsApi(api_client)
    integration_id = 'TGxtSW50ZWdyYXRpb246MTI6YUJjRA==' # str | The unique integration identifier (base64 global ID).

    try:
        # Get an integration
        api_response = api_instance.get_integration(integration_id)
        print("The response of IntegrationsApi->get_integration:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling IntegrationsApi->get_integration: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **integration_id** | **str**| The unique integration identifier (base64 global ID). | 

### Return type

[**LlmIntegration**](LlmIntegration.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An integration object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_integrations**
> ListIntegrationsResponse list_integrations(type=type, space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List integrations

List integrations the user has access to. The response is polymorphic:
each item carries a `type` (and, for `LLM`, `config.provider`) for
client-side discrimination. Use `?type=` to filter to a single type.

Integrations are owned at the account level but carry visibility scopings
(account-wide, organization, or space). `space_id` / `space_name` filter
the list to integrations visible in a given space.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.integration_type import IntegrationType
from arize._generated.api_client.models.list_integrations_response import ListIntegrationsResponse
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
    api_instance = arize._generated.api_client.IntegrationsApi(api_client)
    type = arize._generated.api_client.IntegrationType() # IntegrationType | Filter the list to a single integration type. Omit to list all types. (optional)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List integrations
        api_response = api_instance.list_integrations(type=type, space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of IntegrationsApi->list_integrations:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling IntegrationsApi->list_integrations: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **type** | [**IntegrationType**](.md)| Filter the list to a single integration type. Omit to list all types. | [optional] 
 **space_id** | **str**| Filter search results to a particular space ID | [optional] 
 **space_name** | **str**| Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  | [optional] 
 **name** | **str**| Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, &#x60;name&#x3D;prod&#x60; matches \&quot;production\&quot;, \&quot;my-prod-dataset\&quot;, etc. If omitted, no name filtering is applied and all resources are returned.  | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListIntegrationsResponse**](ListIntegrationsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A paginated, polymorphic list of integrations. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_integration**
> LlmIntegration update_integration(integration_id, body)

Update an integration

Partially update an integration. `type` and `provider` are immutable.
At least one field must be provided.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.llm_integration import LlmIntegration
from arize._generated.api_client.models.update_llm_integration_request import UpdateLlmIntegrationRequest
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
    api_instance = arize._generated.api_client.IntegrationsApi(api_client)
    integration_id = 'TGxtSW50ZWdyYXRpb246MTI6YUJjRA==' # str | The unique integration identifier (base64 global ID).
    body = arize._generated.api_client.UpdateLlmIntegrationRequest() # UpdateLlmIntegrationRequest | Partially update an integration. The body is discriminated by `type`. Omitted fields are left unchanged.  **Payload Requirements** - `type` is **required** (it selects the per-type PATCH shape) and is   immutable: it must match the stored integration's type, otherwise the   request is rejected with 422 (change category by delete + recreate). - At least one updatable field (`name`, `scopings`, or `config`) must be   provided in addition to `type`. - `provider` is immutable. Supplying a value that differs from the stored   integration is rejected with 422. - Envelope and `config` scalar fields deep-merge: omit = keep, explicit   `null` = clear (for nullable fields). - `scopings`, if provided, replaces the existing values. - `config.api_key` may be sent to rotate the key; it is never returned.  **Valid example** ```json {   \"type\": \"LLM\",   \"name\": \"Updated OpenAI\",   \"config\": { \"is_function_calling_enabled\": true } } ```  **Invalid example** (empty body) ```json {} ``` 

    try:
        # Update an integration
        api_response = api_instance.update_integration(integration_id, body)
        print("The response of IntegrationsApi->update_integration:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling IntegrationsApi->update_integration: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **integration_id** | **str**| The unique integration identifier (base64 global ID). | 
 **body** | **UpdateLlmIntegrationRequest**| Partially update an integration. The body is discriminated by &#x60;type&#x60;. Omitted fields are left unchanged.  **Payload Requirements** - &#x60;type&#x60; is **required** (it selects the per-type PATCH shape) and is   immutable: it must match the stored integration&#39;s type, otherwise the   request is rejected with 422 (change category by delete + recreate). - At least one updatable field (&#x60;name&#x60;, &#x60;scopings&#x60;, or &#x60;config&#x60;) must be   provided in addition to &#x60;type&#x60;. - &#x60;provider&#x60; is immutable. Supplying a value that differs from the stored   integration is rejected with 422. - Envelope and &#x60;config&#x60; scalar fields deep-merge: omit &#x3D; keep, explicit   &#x60;null&#x60; &#x3D; clear (for nullable fields). - &#x60;scopings&#x60;, if provided, replaces the existing values. - &#x60;config.api_key&#x60; may be sent to rotate the key; it is never returned.  **Valid example** &#x60;&#x60;&#x60;json {   \&quot;type\&quot;: \&quot;LLM\&quot;,   \&quot;name\&quot;: \&quot;Updated OpenAI\&quot;,   \&quot;config\&quot;: { \&quot;is_function_calling_enabled\&quot;: true } } &#x60;&#x60;&#x60;  **Invalid example** (empty body) &#x60;&#x60;&#x60;json {} &#x60;&#x60;&#x60;  | 

### Return type

[**LlmIntegration**](LlmIntegration.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An integration object. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

