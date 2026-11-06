# arize._generated.api_client.SAMLApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**add_saml_role_mapping**](SAMLApi.md#add_saml_role_mapping) | **POST** /v2/saml-identity-providers/{idp_id}/role-mappings | Add a SAML role mapping
[**create_saml_identity_provider**](SAMLApi.md#create_saml_identity_provider) | **POST** /v2/saml-identity-providers | Create a SAML identity provider
[**delete_saml_identity_provider**](SAMLApi.md#delete_saml_identity_provider) | **DELETE** /v2/saml-identity-providers/{idp_id} | Remove a SAML identity provider
[**get_saml_identity_provider**](SAMLApi.md#get_saml_identity_provider) | **GET** /v2/saml-identity-providers/{idp_id} | Get a SAML identity provider
[**list_saml_identity_providers**](SAMLApi.md#list_saml_identity_providers) | **GET** /v2/saml-identity-providers | List SAML identity providers
[**list_saml_role_mappings**](SAMLApi.md#list_saml_role_mappings) | **GET** /v2/saml-identity-providers/{idp_id}/role-mappings | List SAML role mappings
[**remove_saml_role_mapping**](SAMLApi.md#remove_saml_role_mapping) | **DELETE** /v2/saml-identity-providers/{idp_id}/role-mappings/{mapping_id} | Remove a SAML role mapping
[**update_saml_identity_provider**](SAMLApi.md#update_saml_identity_provider) | **PATCH** /v2/saml-identity-providers/{idp_id} | Update supplied SAML configuration fields
[**update_saml_role_mapping**](SAMLApi.md#update_saml_role_mapping) | **PATCH** /v2/saml-identity-providers/{idp_id}/role-mappings/{mapping_id} | Update supplied mapping fields


# **add_saml_role_mapping**
> SamlRoleMapping add_saml_role_mapping(idp_id, create_saml_role_mapping_request)

Add a SAML role mapping

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Mapping changes are atomic and preserve required login and admin mappings. File-managed configurations are read-only. For custom space roles and project roles, the first matching rule for each resource wins in configuration order. Built-in space roles use the most permissive matching built-in role. New mappings are appended after all existing mappings, at the lowest priority. For existing users, changes to stored role assignments apply at their next successful SAML login only when sync_user_roles is true. Mapping changes do not revoke current sessions or API keys.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_saml_role_mapping_request import CreateSamlRoleMappingRequest
from arize._generated.api_client.models.saml_role_mapping import SamlRoleMapping
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.
    create_saml_role_mapping_request = {"attributes":{"groups":"admins"},"is_account_admin":true,"organization":null,"space_roles":[]} # CreateSamlRoleMappingRequest | 

    try:
        # Add a SAML role mapping
        api_response = api_instance.add_saml_role_mapping(idp_id, create_saml_role_mapping_request)
        print("The response of SAMLApi->add_saml_role_mapping:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->add_saml_role_mapping: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 
 **create_saml_role_mapping_request** | [**CreateSamlRoleMappingRequest**](CreateSamlRoleMappingRequest.md)|  | 

### Return type

[**SamlRoleMapping**](SamlRoleMapping.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Successful response. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_saml_identity_provider**
> SamlIdentityProvider create_saml_identity_provider(create_saml_identity_provider_request)

Create a SAML identity provider

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Requests use the authenticated caller's account. File-managed configurations are read-only. Default login requires all four default fields. With default login disabled, supply initial role mappings and omit defaults. Domain verification flags are not accepted. Invalid metadata or an unreachable metadata URL returns 400.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_saml_identity_provider_request import CreateSamlIdentityProviderRequest
from arize._generated.api_client.models.saml_identity_provider import SamlIdentityProvider
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    create_saml_identity_provider_request = {"email_domains":["example.com"],"metadata_xml":"<EntityDescriptor xmlns=\"urn:oasis:names:tc:SAML:2.0:metadata\" entityID=\"https://login.example.com\"><IDPSSODescriptor protocolSupportEnumeration=\"urn:oasis:names:tc:SAML:2.0:protocol\"><SingleSignOnService Binding=\"urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect\" Location=\"https://login.example.com/sso\"/></IDPSSODescriptor></EntityDescriptor>","enforce_saml":false,"sync_user_roles":true,"sign_authn":true,"allow_login_with_defaults":false,"role_mappings":[{"attributes":{"groups":"admins"},"is_account_admin":true,"organization":null,"space_roles":[]}]} # CreateSamlIdentityProviderRequest | 

    try:
        # Create a SAML identity provider
        api_response = api_instance.create_saml_identity_provider(create_saml_identity_provider_request)
        print("The response of SAMLApi->create_saml_identity_provider:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->create_saml_identity_provider: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_saml_identity_provider_request** | [**CreateSamlIdentityProviderRequest**](CreateSamlIdentityProviderRequest.md)|  | 

### Return type

[**SamlIdentityProvider**](SamlIdentityProvider.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Successful response. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_saml_identity_provider**
> delete_saml_identity_provider(idp_id)

Remove a SAML identity provider

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Requests use the authenticated caller's account. File-managed configurations are read-only. Disable enforcement before deletion. Deletion that would restore a hidden file configuration returns 409. Users and role assignments are retained.

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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.

    try:
        # Remove a SAML identity provider
        api_instance.delete_saml_identity_provider(idp_id)
    except Exception as e:
        print("Exception when calling SAMLApi->delete_saml_identity_provider: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 

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
**204** | Removed successfully. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_saml_identity_provider**
> SamlIdentityProvider get_saml_identity_provider(idp_id)

Get a SAML identity provider

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Requests use the authenticated caller's account. File-managed configurations are read-only.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.saml_identity_provider import SamlIdentityProvider
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.

    try:
        # Get a SAML identity provider
        api_response = api_instance.get_saml_identity_provider(idp_id)
        print("The response of SAMLApi->get_saml_identity_provider:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->get_saml_identity_provider: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 

### Return type

[**SamlIdentityProvider**](SamlIdentityProvider.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_saml_identity_providers**
> ListSamlIdentityProvidersResponse list_saml_identity_providers(limit=limit, cursor=cursor)

List SAML identity providers

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Requests use the authenticated caller's account. File-managed configurations are read-only. Returns summary fields only. Read a specific identity provider for metadata, defaults and role mappings.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_saml_identity_providers_response import ListSamlIdentityProvidersResponse
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    limit = 100 # int | Maximum number of configurations to return. (optional) (default to 100)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List SAML identity providers
        api_response = api_instance.list_saml_identity_providers(limit=limit, cursor=cursor)
        print("The response of SAMLApi->list_saml_identity_providers:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->list_saml_identity_providers: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **limit** | **int**| Maximum number of configurations to return. | [optional] [default to 100]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListSamlIdentityProvidersResponse**](ListSamlIdentityProvidersResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **list_saml_role_mappings**
> ListSamlRoleMappingsResponse list_saml_role_mappings(idp_id, limit=limit, cursor=cursor)

List SAML role mappings

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Mapping changes are atomic and preserve required login and admin mappings. File-managed configurations are read-only. Mappings are returned in configuration order. For custom space roles and project roles, the first matching rule for each resource wins in configuration order. Built-in space roles use the most permissive matching built-in role.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.list_saml_role_mappings_response import ListSamlRoleMappingsResponse
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.
    limit = 100 # int | Maximum number of configurations to return. (optional) (default to 100)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List SAML role mappings
        api_response = api_instance.list_saml_role_mappings(idp_id, limit=limit, cursor=cursor)
        print("The response of SAMLApi->list_saml_role_mappings:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->list_saml_role_mappings: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 
 **limit** | **int**| Maximum number of configurations to return. | [optional] [default to 100]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**ListSamlRoleMappingsResponse**](ListSamlRoleMappingsResponse.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **remove_saml_role_mapping**
> remove_saml_role_mapping(idp_id, mapping_id)

Remove a SAML role mapping

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Mapping changes are atomic and preserve required login and admin mappings. File-managed configurations are read-only. For custom space roles and project roles, the first matching rule for each resource wins in configuration order. Built-in space roles use the most permissive matching built-in role. Removing and re-adding a mapping moves it to the end and can change effective roles for users who match multiple rules. For existing users, changes to stored role assignments apply at their next successful SAML login only when sync_user_roles is true. Mapping changes do not revoke current sessions or API keys.

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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.
    mapping_id = 'mapping_id_example' # str | Mapping ID returned by list or add.

    try:
        # Remove a SAML role mapping
        api_instance.remove_saml_role_mapping(idp_id, mapping_id)
    except Exception as e:
        print("Exception when calling SAMLApi->remove_saml_role_mapping: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 
 **mapping_id** | **str**| Mapping ID returned by list or add. | 

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
**204** | Removed successfully. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_saml_identity_provider**
> SamlIdentityProvider update_saml_identity_provider(idp_id, update_saml_identity_provider_request)

Update supplied SAML configuration fields

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Requests use the authenticated caller's account. File-managed configurations are read-only. Invalid metadata or an unreachable metadata URL returns 400.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.saml_identity_provider import SamlIdentityProvider
from arize._generated.api_client.models.update_saml_identity_provider_request import UpdateSamlIdentityProviderRequest
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.
    update_saml_identity_provider_request = {"sync_user_roles":false} # UpdateSamlIdentityProviderRequest | 

    try:
        # Update supplied SAML configuration fields
        api_response = api_instance.update_saml_identity_provider(idp_id, update_saml_identity_provider_request)
        print("The response of SAMLApi->update_saml_identity_provider:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->update_saml_identity_provider: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 
 **update_saml_identity_provider_request** | [**UpdateSamlIdentityProviderRequest**](UpdateSamlIdentityProviderRequest.md)|  | 

### Return type

[**SamlIdentityProvider**](SamlIdentityProvider.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **update_saml_role_mapping**
> SamlRoleMapping update_saml_role_mapping(idp_id, mapping_id, update_saml_role_mapping_request)

Update supplied mapping fields

Warning: This API is alpha and can change without notice. Requires Enterprise account-admin access. Mapping changes are atomic and preserve required login and admin mappings. File-managed configurations are read-only. For custom space roles and project roles, the first matching rule for each resource wins in configuration order. Built-in space roles use the most permissive matching built-in role. Updating a mapping preserves its position. For existing users, changes to stored role assignments apply at their next successful SAML login only when sync_user_roles is true. Mapping changes do not revoke current sessions or API keys.

### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.saml_role_mapping import SamlRoleMapping
from arize._generated.api_client.models.update_saml_role_mapping_request import UpdateSamlRoleMappingRequest
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
    api_instance = arize._generated.api_client.SAMLApi(api_client)
    idp_id = 'idp_id_example' # str | Identity provider ID returned by the list operation.
    mapping_id = 'mapping_id_example' # str | Mapping ID returned by list or add.
    update_saml_role_mapping_request = {"attributes":{"groups":"admins-renamed"}} # UpdateSamlRoleMappingRequest | 

    try:
        # Update supplied mapping fields
        api_response = api_instance.update_saml_role_mapping(idp_id, mapping_id, update_saml_role_mapping_request)
        print("The response of SAMLApi->update_saml_role_mapping:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling SAMLApi->update_saml_role_mapping: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **idp_id** | **str**| Identity provider ID returned by the list operation. | 
 **mapping_id** | **str**| Mapping ID returned by list or add. | 
 **update_saml_role_mapping_request** | [**UpdateSamlRoleMappingRequest**](UpdateSamlRoleMappingRequest.md)|  | 

### Return type

[**SamlRoleMapping**](SamlRoleMapping.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful response. |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**422** | Unprocessable entity |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

