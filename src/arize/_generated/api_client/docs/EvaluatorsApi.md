# arize._generated.api_client.EvaluatorsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**create_evaluator**](EvaluatorsApi.md#create_evaluator) | **POST** /v2/evaluators | Create evaluator
[**create_evaluator_version**](EvaluatorsApi.md#create_evaluator_version) | **POST** /v2/evaluators/{evaluator_id}/versions | Create evaluator version
[**delete_evaluator**](EvaluatorsApi.md#delete_evaluator) | **DELETE** /v2/evaluators/{evaluator_id} | Delete evaluator
[**get_evaluator**](EvaluatorsApi.md#get_evaluator) | **GET** /v2/evaluators/{evaluator_id} | Get evaluator
[**get_evaluator_version**](EvaluatorsApi.md#get_evaluator_version) | **GET** /v2/evaluator-versions/{version_id} | Get evaluator version
[**list_evaluator_versions**](EvaluatorsApi.md#list_evaluator_versions) | **GET** /v2/evaluators/{evaluator_id}/versions | List evaluator versions
[**list_evaluators**](EvaluatorsApi.md#list_evaluators) | **GET** /v2/evaluators | List evaluators
[**update_evaluator**](EvaluatorsApi.md#update_evaluator) | **PATCH** /v2/evaluators/{evaluator_id} | Update evaluator


# **create_evaluator**
> EvaluatorWithVersion create_evaluator(create_evaluator_request)

Create evaluator

Creates a new evaluator with an initial version.

**Payload Requirements**
- The evaluator `name` must be unique within the given space.
- `type` (top-level) selects the evaluator kind: `TEMPLATE` or `CODE`.
  With `TEMPLATE`, provide `version.template_config`.
  With `CODE`, provide `version.code_config` — where `code_config.type` is `MANAGED` or `CUSTOM` (a separate discriminator *within* `code_config`, independent of the top-level `type: CODE`).
- For template evaluators: `version.template_config.name` is the eval column name; must match `^[a-zA-Z0-9_\s\-&()]+$`.
- For template evaluators: `version.template_config.template` is the prompt template; use `{variable}` for placeholders (f-string format, e.g. `{input}`, `{output}`).
- For template evaluators: `version.template_config.classification_choices` maps choice labels to numeric scores (e.g. `{"relevant": 1, "irrelevant": 0}`). When omitted, the evaluator produces freeform output.
- For code evaluators: see `CodeConfig` — managed evaluators (`code_config.type: MANAGED`) use `managed_evaluator` and `variables`; custom evaluators (`code_config.type: CUSTOM`) use `code`, optional `imports`, and `variables`.
- System-managed fields (`id`, `created_at`, `updated_at`, `created_by_user_id`) are rejected on input.

**Valid example** (template evaluator)
```json
{
  "name": "Hallucination Detector",
  "space_id": "U3BhY2U6MTpWNEth",
  "type": "TEMPLATE",
  "version": {
    "commit_message": "Initial version",
    "template_config": {
      "name": "hallucination",
      "template": "Given the input: {input}\nand the output: {output}\nIs the output a hallucination?",
      "include_explanations": true,
      "use_function_calling_if_available": true,
      "classification_choices": {"hallucinated": 0, "factual": 1},
      "llm_config": {
        "ai_integration_id": "TGxtSW50ZWdyYXRpb246MTI6YUJjRA==",
        "model_name": "gpt-4o",
        "invocation_parameters": {"temperature": 0},
        "provider_parameters": {}
      }
    }
  }
}
```

**Invalid example** (type/config mismatch — `TEMPLATE` type with `code_config`)
```json
{
  "name": "Bad Evaluator",
  "space_id": "U3BhY2U6MTpWNEth",
  "type": "TEMPLATE",
  "version": {
    "commit_message": "Wrong config",
    "code_config": {
      "type": "CUSTOM",
      "name": "my_eval",
      "code": "class Evaluator: ...",
      "variables": ["input"]
    }
  }
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_evaluator_request import CreateEvaluatorRequest
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
    create_evaluator_request = {"space_id":"U3BhY2U6NDkzOkJaSkc=","name":"Hallucination Eval","description":"Detects hallucinated content in LLM responses","type":"TEMPLATE","version":{"commit_message":"Initial version","template_config":{"name":"hallucination","template":"You are an evaluation assistant. Given the following input and output, determine if the output contains hallucinated content.\n\nInput: {input}\nOutput: {output}\nReference: {reference}","include_explanations":true,"use_function_calling_if_available":true,"classification_choices":{"hallucinated":0,"factual":1},"direction":"MAXIMIZE","data_granularity":"SPAN","llm_config":{"ai_integration_id":"TGxtSW50ZWdyYXRpb246MTI6YUJjRA==","model_name":"gpt-4o","invocation_parameters":{"temperature":0},"provider_parameters":{}}}}} # CreateEvaluatorRequest | Body containing evaluator creation parameters with an initial version.  Only `type: TEMPLATE` and `type: CODE` are currently accepted on creation. 

    try:
        # Create evaluator
        api_response = api_instance.create_evaluator(create_evaluator_request)
        print("The response of EvaluatorsApi->create_evaluator:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->create_evaluator: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_evaluator_request** | [**CreateEvaluatorRequest**](CreateEvaluatorRequest.md)| Body containing evaluator creation parameters with an initial version.  Only &#x60;type: TEMPLATE&#x60; and &#x60;type: CODE&#x60; are currently accepted on creation.  | 

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
**201** | Returns an evaluator with a resolved version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_evaluator_version**
> EvaluatorVersion create_evaluator_version(evaluator_id, create_evaluator_version_request)

Create evaluator version

Create a new version of an existing evaluator. The new version becomes the latest
version immediately (versioning is append-only).

**Payload Requirements**
- `commit_message` describes the changes in this version.
- Provide either `template_config` or `code_config` to match the evaluator's `type`.
  `code_config.type` is a separate inner discriminator (`MANAGED` or `CUSTOM`) and is unrelated to the top-level `type`.
  Schema and constraints match Create Evaluator.

**Valid example** (template version)
```json
{
  "commit_message": "Improve prompt template for better accuracy",
  "template_config": {
    "name": "hallucination",
    "template": "Given the input: {input}\nand output: {output}\nIs the output a hallucination? Explain your reasoning.",
    "include_explanations": true,
    "use_function_calling_if_available": true,
    "classification_choices": {"hallucinated": 0, "factual": 1},
    "llm_config": {
      "ai_integration_id": "TGxtSW50ZWdyYXRpb246MTI6YUJjRA==",
      "model_name": "gpt-4o",
      "invocation_parameters": {"temperature": 0},
      "provider_parameters": {}
    }
  }
}
```

**Invalid example** (missing required `commit_message`)
```json
{
  "template_config": {
    "name": "hallucination",
    "template": "Is this a hallucination?",
    "include_explanations": false,
    "use_function_calling_if_available": false,
    "llm_config": {
      "ai_integration_id": "TGxtSW50ZWdyYXRpb246MTI6YUJjRA==",
      "model_name": "gpt-4o",
      "invocation_parameters": {},
      "provider_parameters": {}
    }
  }
}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_evaluator_version_request import CreateEvaluatorVersionRequest
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
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The unique evaluator identifier (base64)
    create_evaluator_version_request = {"commit_message":"Improve template wording","template_config":{"name":"hallucination","template":"Evaluate whether the output is factually grounded.\n\nInput: {input}\nOutput: {output}","include_explanations":true,"use_function_calling_if_available":true,"classification_choices":{"hallucinated":0,"factual":1},"direction":"MAXIMIZE","data_granularity":"SPAN","llm_config":{"ai_integration_id":"TGxtSW50ZWdyYXRpb246MTI6YUJjRA==","model_name":"gpt-4o","invocation_parameters":{"temperature":0},"provider_parameters":{}}}} # CreateEvaluatorVersionRequest | Body containing evaluator version creation parameters

    try:
        # Create evaluator version
        api_response = api_instance.create_evaluator_version(evaluator_id, create_evaluator_version_request)
        print("The response of EvaluatorsApi->create_evaluator_version:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->create_evaluator_version: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The unique evaluator identifier (base64) | 
 **create_evaluator_version_request** | [**CreateEvaluatorVersionRequest**](CreateEvaluatorVersionRequest.md)| Body containing evaluator version creation parameters | 

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
**201** | Returns an evaluator version |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_evaluator**
> delete_evaluator(evaluator_id)

Delete evaluator

Deletes an evaluator and all its versions. This operation is irreversible.

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
    api_instance = arize._generated.api_client.EvaluatorsApi(api_client)
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The unique evaluator identifier (base64)

    try:
        # Delete evaluator
        api_instance.delete_evaluator(evaluator_id)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->delete_evaluator: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The unique evaluator identifier (base64) | 

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

# **get_evaluator**
> EvaluatorWithVersion get_evaluator(evaluator_id, version_id=version_id)

Get evaluator

Returns an evaluator and a resolved version. By default, the latest version
is included. Use the version_id query parameter to resolve a specific version.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The unique evaluator identifier (base64)
    version_id = 'RXZhbHVhdG9yVmVyc2lvbjoxMjM0NQ==' # str | Return the evaluator with this specific version (base64 identifier (base64)). If omitted, returns the latest version. (optional)

    try:
        # Get evaluator
        api_response = api_instance.get_evaluator(evaluator_id, version_id=version_id)
        print("The response of EvaluatorsApi->get_evaluator:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->get_evaluator: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The unique evaluator identifier (base64) | 
 **version_id** | **str**| Return the evaluator with this specific version (base64 identifier (base64)). If omitted, returns the latest version. | [optional] 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_evaluator_version**
> EvaluatorVersion get_evaluator_version(version_id)

Get evaluator version

Get a specific evaluator version by its unique identifier.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


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
    version_id = 'RXZhbHVhdG9yVmVyc2lvbjoxMjM0NQ==' # str | The unique evaluator version identifier (base64)

    try:
        # Get evaluator version
        api_response = api_instance.get_evaluator_version(version_id)
        print("The response of EvaluatorsApi->get_evaluator_version:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->get_evaluator_version: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **version_id** | **str**| The unique evaluator version identifier (base64) | 

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_evaluator_versions**
> ListEvaluatorVersionsResponse list_evaluator_versions(evaluator_id, limit=limit, cursor=cursor)

List evaluator versions

List all versions of an evaluator with cursor-based pagination.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_evaluator_versions_response import ListEvaluatorVersionsResponse
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
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The unique evaluator identifier (base64)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List evaluator versions
        api_response = api_instance.list_evaluator_versions(evaluator_id, limit=limit, cursor=cursor)
        print("The response of EvaluatorsApi->list_evaluator_versions:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->list_evaluator_versions: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The unique evaluator identifier (base64) | 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListEvaluatorVersionsResponse**](ListEvaluatorVersionsResponse.md)

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

# **list_evaluators**
> ListEvaluatorsResponse list_evaluators(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)

List evaluators

List evaluators the user has access to, sorted by update date (most recent first).

When `space_id` is provided, results are limited to that space. When omitted,
evaluators from all permitted spaces are returned.

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_evaluators_response import ListEvaluatorsResponse
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
        api_response = api_instance.list_evaluators(space_id=space_id, space_name=space_name, name=name, limit=limit, cursor=cursor)
        print("The response of EvaluatorsApi->list_evaluators:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->list_evaluators: %s\n" % e)
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

[**ListEvaluatorsResponse**](ListEvaluatorsResponse.md)

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
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_evaluator**
> Evaluator update_evaluator(evaluator_id, update_evaluator_request)

Update evaluator

Update an evaluator's metadata. At least one field must be provided.
Omitted fields are left unchanged.

**Payload Requirements**
- At least one of `name` or `description` must be provided.
- `name`, if provided, must be unique within the space.
- System-managed fields (`id`, `type`, `space_id`, `created_at`, `updated_at`, `created_by_user_id`) cannot be modified.

**Valid example**
```json
{
  "name": "Hallucination Detector v2",
  "description": "Updated evaluator for production hallucination checks"
}
```

**Invalid example** (no updatable fields provided)
```json
{}
```

<Note>This endpoint is in beta, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Note>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.evaluator import Evaluator
from arize._generated.api_client.models.update_evaluator_request import UpdateEvaluatorRequest
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
    evaluator_id = 'RXZhbHVhdG9yOjEyMzQ1' # str | The unique evaluator identifier (base64)
    update_evaluator_request = {"name":"Updated Evaluator Name","description":"Updated description"} # UpdateEvaluatorRequest | Body containing evaluator update parameters

    try:
        # Update evaluator
        api_response = api_instance.update_evaluator(evaluator_id, update_evaluator_request)
        print("The response of EvaluatorsApi->update_evaluator:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling EvaluatorsApi->update_evaluator: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **evaluator_id** | **str**| The unique evaluator identifier (base64) | 
 **update_evaluator_request** | [**UpdateEvaluatorRequest**](UpdateEvaluatorRequest.md)| Body containing evaluator update parameters | 

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
**200** | An evaluator object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

