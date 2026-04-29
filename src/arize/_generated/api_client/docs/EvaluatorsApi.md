# arize._generated.api_client.EvaluatorsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**evaluator_versions_create**](EvaluatorsApi.md#evaluator_versions_create) | **POST** /v2/evaluators/{evaluator_id}/versions | Create evaluator version
[**evaluator_versions_get**](EvaluatorsApi.md#evaluator_versions_get) | **GET** /v2/evaluator-versions/{version_id} | Get evaluator version
[**evaluator_versions_list**](EvaluatorsApi.md#evaluator_versions_list) | **GET** /v2/evaluators/{evaluator_id}/versions | List evaluator versions
[**evaluators_create**](EvaluatorsApi.md#evaluators_create) | **POST** /v2/evaluators | Create evaluator
[**evaluators_delete**](EvaluatorsApi.md#evaluators_delete) | **DELETE** /v2/evaluators/{evaluator_id} | Delete evaluator
[**evaluators_get**](EvaluatorsApi.md#evaluators_get) | **GET** /v2/evaluators/{evaluator_id} | Get evaluator
[**evaluators_list**](EvaluatorsApi.md#evaluators_list) | **GET** /v2/evaluators | List evaluators
[**evaluators_update**](EvaluatorsApi.md#evaluators_update) | **PATCH** /v2/evaluators/{evaluator_id} | Update evaluator


# **evaluator_versions_create**
> EvaluatorVersion evaluator_versions_create(evaluator_id, evaluator_version_create)

Create evaluator version

Create a new version of an existing evaluator. The new version becomes the latest
version immediately (versioning is append-only).

**Payload Requirements**
- `commit_message` describes the changes in this version.
- Provide either `template_config` or `code_config` to match the evaluator's `type`.
  `code_config.type` is a separate inner discriminator (`managed` or `custom`) and is unrelated to the top-level `type`.
  Schema and constraints match Create Evaluator.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator_version import EvaluatorVersion
from arize._generated.api_client.models.evaluator_version_create import EvaluatorVersionCreate
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The evaluator global ID (base64)
    evaluator_version_create = {"commit_message":"Improve template wording","template_config":{"name":"hallucination","template":"Evaluate whether the output is factually grounded.\n\nInput: {input}\nOutput: {output}","include_explanations":true,"use_function_calling_if_available":true,"classification_choices":{"hallucinated":0,"factual":1},"direction":"maximize","data_granularity":"span","llm_config":{"ai_integration_id":"TGxtSW50ZWdyYXRpb246MTI6YUJjRA==","model_name":"gpt-4o","invocation_parameters":{"temperature":0},"provider_parameters":{}}}} # EvaluatorVersionCreate | Body containing evaluator version creation parameters

    try:
        # Create evaluator version
        api_response = api_instance.evaluator_versions_create(evaluator_id, evaluator_version_create)
        print("The response of EvaluatorsApi->evaluator_versions_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluator_versions_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The evaluator global ID (base64) | 
 **evaluator_version_create** | [**EvaluatorVersionCreate**](EvaluatorVersionCreate.md)| Body containing evaluator version creation parameters | 

### Return type

[**EvaluatorVersion**](EvaluatorVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created evaluator version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluator_versions_get**
> EvaluatorVersion evaluator_versions_get(version_id)

Get evaluator version

Get a specific evaluator version by its global ID.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator_version import EvaluatorVersion
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    version_id = 'RXZhbHVhdG9yVmVyc2lvbjoxMjM0NQ==' # str | The evaluator version global ID (base64)

    try:
        # Get evaluator version
        api_response = api_instance.evaluator_versions_get(version_id)
        print("The response of EvaluatorsApi->evaluator_versions_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluator_versions_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The evaluator version global ID (base64) | 

### Return type

[**EvaluatorVersion**](EvaluatorVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns an evaluator version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluator_versions_list**
> EvaluatorVersionsList200Response evaluator_versions_list(evaluator_id, limit=limit, cursor=cursor)

List evaluator versions

List all versions of an evaluator with cursor-based pagination.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator_versions_list200_response import EvaluatorVersionsList200Response
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The evaluator global ID (base64)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List evaluator versions
        api_response = api_instance.evaluator_versions_list(evaluator_id, limit=limit, cursor=cursor)
        print("The response of EvaluatorsApi->evaluator_versions_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluator_versions_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The evaluator global ID (base64) | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**EvaluatorVersionsList200Response**](EvaluatorVersionsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of evaluator version objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluators_create**
> EvaluatorWithVersion evaluators_create(evaluators_create_request)

Create evaluator

Creates a new evaluator with an initial version.

**Payload Requirements**
- The evaluator `name` must be unique within the given space.
- `type` (top-level) selects the evaluator kind: `template` or `code`.
  With `template`, provide `version.template_config`.
  With `code`, provide `version.code_config` — where `code_config.type` is `managed` or `custom` (a separate discriminator *within* `code_config`, independent of the top-level `type: code`).
- For template evaluators: `version.template_config.name` is the eval column name; must match `^[a-zA-Z0-9_\s\-&()]+$`.
- For template evaluators: `version.template_config.template` is the prompt template; use `{variable}` for placeholders (f-string format, e.g. `{input}`, `{output}`).
- For template evaluators: `version.template_config.classification_choices` maps choice labels to numeric scores (e.g. `{"relevant": 1, "irrelevant": 0}`). When omitted, the evaluator produces freeform output.
- For code evaluators: see `CodeConfig` — managed evaluators (`code_config.type: managed`) use `managed_evaluator` and `variables`; custom evaluators (`code_config.type: custom`) use `code`, optional `imports`, and `variables`.
- System-managed fields (`id`, `created_at`, `updated_at`, `created_by_user_id`) are rejected on input.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator_with_version import EvaluatorWithVersion
from arize._generated.api_client.models.evaluators_create_request import EvaluatorsCreateRequest
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluators_create_request = {"space_id":"U3BhY2U6NDkzOkJaSkc=","name":"Hallucination Eval","description":"Detects hallucinated content in LLM responses","type":"template","version":{"commit_message":"Initial version","template_config":{"name":"hallucination","template":"You are an evaluation assistant. Given the following input and output, determine if the output contains hallucinated content.\n\nInput: {input}\nOutput: {output}\nReference: {reference}","include_explanations":true,"use_function_calling_if_available":true,"classification_choices":{"hallucinated":0,"factual":1},"direction":"maximize","data_granularity":"span","llm_config":{"ai_integration_id":"TGxtSW50ZWdyYXRpb246MTI6YUJjRA==","model_name":"gpt-4o","invocation_parameters":{"temperature":0},"provider_parameters":{}}}}} # EvaluatorsCreateRequest | Body containing evaluator creation parameters with an initial version

    try:
        # Create evaluator
        api_response = api_instance.evaluators_create(evaluators_create_request)
        print("The response of EvaluatorsApi->evaluators_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluators_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluators_create_request** | [**EvaluatorsCreateRequest**](EvaluatorsCreateRequest.md)| Body containing evaluator creation parameters with an initial version | 

### Return type

[**EvaluatorWithVersion**](EvaluatorWithVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Returns the created evaluator with its initial version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluators_delete**
> evaluators_delete(evaluator_id)

Delete evaluator

Deletes an evaluator and all its versions. This operation is irreversible.

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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The evaluator global ID (base64)

    try:
        # Delete evaluator
        api_instance.evaluators_delete(evaluator_id)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluators_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The evaluator global ID (base64) | 

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
**204** | Evaluator deleted successfully |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluators_get**
> EvaluatorWithVersion evaluators_get(evaluator_id, version_id=version_id)

Get evaluator

Returns an evaluator and a resolved version. By default, the latest version
is included. Use the version_id query parameter to resolve a specific version.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator_with_version import EvaluatorWithVersion
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The evaluator global ID (base64)
    version_id = 'RXZhbHVhdG9yVmVyc2lvbjoxMjM0NQ==' # str | Return the evaluator with this specific version (base64 global ID). If omitted, returns the latest version. (optional)

    try:
        # Get evaluator
        api_response = api_instance.evaluators_get(evaluator_id, version_id=version_id)
        print("The response of EvaluatorsApi->evaluators_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluators_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The evaluator global ID (base64) | 
 **version_id** | **str**| Return the evaluator with this specific version (base64 global ID). If omitted, returns the latest version. | [optional] 

### Return type

[**EvaluatorWithVersion**](EvaluatorWithVersion.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns an evaluator with a resolved version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluators_list**
> EvaluatorsList200Response evaluators_list(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List evaluators

List evaluators the user has access to, sorted by update date (most recent first).

When `space_id` is provided, results are limited to that space. When omitted,
evaluators from all permitted spaces are returned.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluators_list200_response import EvaluatorsList200Response
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    space_id = 'U3BhY2U6MTIzNDU=' # str | Filter search results to a particular space ID (optional)
    space_name = 'my-space' # str | Case-insensitive substring filter on the space name. Narrows results to resources in spaces whose name contains the given string. If omitted, no space name filtering is applied and all resources are returned.  (optional)
    name = 'production' # str | Case-insensitive substring filter on the resource name. Returns only resources whose name contains the given string. For example, `name=prod` matches \"production\", \"my-prod-dataset\", etc. If omitted, no name filtering is applied and all resources are returned.  (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List evaluators
        api_response = api_instance.evaluators_list(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of EvaluatorsApi->evaluators_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluators_list: %s\n" % e)
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

[**EvaluatorsList200Response**](EvaluatorsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of evaluator objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **evaluators_update**
> Evaluator evaluators_update(evaluator_id, evaluators_update_request)

Update evaluator

Update an evaluator's metadata. At least one field must be provided.
Omitted fields are left unchanged.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator import Evaluator
from arize._generated.api_client.models.evaluators_update_request import EvaluatorsUpdateRequest
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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The evaluator global ID (base64)
    evaluators_update_request = {"name":"Updated Evaluator Name","description":"Updated description"} # EvaluatorsUpdateRequest | Body containing evaluator update parameters

    try:
        # Update evaluator
        api_response = api_instance.evaluators_update(evaluator_id, evaluators_update_request)
        print("The response of EvaluatorsApi->evaluators_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->evaluators_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The evaluator global ID (base64) | 
 **evaluators_update_request** | [**EvaluatorsUpdateRequest**](EvaluatorsUpdateRequest.md)| Body containing evaluator update parameters | 

### Return type

[**Evaluator**](Evaluator.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns the updated evaluator |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

