# arize._generated.api_client.UsersApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**users_create**](UsersApi.md#users_create) | **POST** /v2/users | Create a user
[**users_delete**](UsersApi.md#users_delete) | **DELETE** /v2/users/{user_id} | Delete a user
[**users_get**](UsersApi.md#users_get) | **GET** /v2/users/{user_id} | Get a user
[**users_list**](UsersApi.md#users_list) | **GET** /v2/users | List users
[**users_password_reset**](UsersApi.md#users_password_reset) | **POST** /v2/users/{user_id}/reset-password | Trigger a password-reset email for a user
[**users_resend_invitation**](UsersApi.md#users_resend_invitation) | **POST** /v2/users/{user_id}/resend-invitation | Resend a user invitation
[**users_update**](UsersApi.md#users_update) | **PATCH** /v2/users/{user_id} | Update a user


# **users_create**
> User users_create(create_user_request)

Create a user

Create a new account user with explicit invite control.

**Invite modes**
- `none` — add the user directly with no invitation (for SSO-only accounts). The user
  is immediately active and can log in via the configured identity provider.
- `email_link` — create an `invited` invitation and send the user an email with a
  verification link to complete registration.
- `temporary_password` — create an `invited` invitation with a temporary password
  (returned once in the response). The user must reset it on first login.

**Idempotency on `email`** (applies when `invite_mode != "none"`)

| Existing state | Behavior | Response |
| --- | --- | --- |
| No prior invitation | Create a new `invited` invitation | `201 Created` |
| `invited` (not yet accepted) | Return the existing invitation as-is; do not resend | `200 OK` |
| `active` | Email belongs to an existing member | `409 Conflict` |
| `expired` | Create a new `invited` invitation | `201 Created` |
| `inactive` | User has been deactivated and cannot be re-invited | `409 Conflict` |

When `invite_mode` is `none` and the email already belongs to an active account member,
the request returns `409 Conflict`.

**Payload requirements**
- `name` — required, 1–255 characters
- `email` — required, must be a valid email address; used as the idempotency key
- `role` — required, one of `admin`, `member`, `annotator`; sets the account-level role
- `invite_mode` — required, one of `none`, `email_link`, `temporary_password`

Requires account admin role or USER_CREATE permission.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.create_user_request import CreateUserRequest
from arize._generated.api_client.models.user import User
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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    create_user_request = {"name":"Jane Smith","email":"jane.smith@example.com","role":{"type":"predefined","name":"member"},"invite_mode":"email_link"} # CreateUserRequest | Body containing user creation parameters and invite control.

    try:
        # Create a user
        api_response = api_instance.users_create(create_user_request)
        print("The response of UsersApi->users_create:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UsersApi->users_create: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **create_user_request** | [**CreateUserRequest**](CreateUserRequest.md)| Body containing user creation parameters and invite control. | 

### Return type

[**User**](User.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An account user object |  -  |
**201** | User created successfully |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**409** | Resource conflict |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **users_delete**
> users_delete(user_id)

Delete a user

Permanently block a user from the account. The user's status is set to
`inactive` and they can no longer log in. The operation cascades to:
- Organization memberships
- Space memberships
- User API keys
- Role bindings

Blocked users cannot be re-invited via the create endpoint — `inactive`
is a terminal state. Callers cannot delete themselves. The operation is
idempotent — blocking an already-inactive user returns 204.

Requires account admin role or USER_DELETE permission.

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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user

    try:
        # Delete a user
        api_instance.users_delete(user_id)
    except Exception as e:
        print("Exception when calling UsersApi->users_delete: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **user_id** | **str**| The unique identifier of the user | 

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
**204** | User successfully removed from the account |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **users_get**
> User users_get(user_id)

Get a user

Get a specific user by their ID.

Requires account admin role, account member role, or USER_READ permission at the account level.

Returns 404 if the user does not exist, does not belong to the caller's account, or the caller lacks read permission.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.user import User
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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user

    try:
        # Get a user
        api_response = api_instance.users_get(user_id)
        print("The response of UsersApi->users_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UsersApi->users_get: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **user_id** | **str**| The unique identifier of the user | 

### Return type

[**User**](User.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An account user object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **users_list**
> UsersList200Response users_list(limit=limit, cursor=cursor, email=email, status=status)

List users

List users in the account with cursor-based pagination. Results are sorted by
creation date ascending (oldest first).

Requires account admin role, account member role, or USER_READ permission at the
account level.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.user_status import UserStatus
from arize._generated.api_client.models.users_list200_response import UsersList200Response
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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)
    email = 'jane@example.com' # str | Filter users by email address (case-insensitive partial match, up to 255 characters). Results are scoped to users visible to the caller.  (optional)
    status = [arize._generated.api_client.UserStatus()] # List[UserStatus] | Filter users by account status. When omitted, `active`, `invited`, and `expired` users are returned. Can be specified multiple times to filter by multiple statuses (e.g., `?status=active&status=invited`).  (optional)

    try:
        # List users
        api_response = api_instance.users_list(limit=limit, cursor=cursor, email=email, status=status)
        print("The response of UsersApi->users_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UsersApi->users_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 
 **email** | **str**| Filter users by email address (case-insensitive partial match, up to 255 characters). Results are scoped to users visible to the caller.  | [optional] 
 **status** | [**List[UserStatus]**](UserStatus.md)| Filter users by account status. When omitted, &#x60;active&#x60;, &#x60;invited&#x60;, and &#x60;expired&#x60; users are returned. Can be specified multiple times to filter by multiple statuses (e.g., &#x60;?status&#x3D;active&amp;status&#x3D;invited&#x60;).  | [optional] 

### Return type

[**UsersList200Response**](UsersList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Returns a list of account user objects |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **users_password_reset**
> users_password_reset(user_id)

Trigger a password-reset email for a user

Generates a reset token and sends the user a password-reset email with a 30-minute link.

- Requires account admin role or USER_UPDATE permission.
- Returns 400 if the target user authenticates via SSO/SAML or has not
  yet verified their account (no password hash to key the token against).

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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user

    try:
        # Trigger a password-reset email for a user
        api_instance.users_password_reset(user_id)
    except Exception as e:
        print("Exception when calling UsersApi->users_password_reset: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **user_id** | **str**| The unique identifier of the user | 

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
**204** | Password-reset email sent successfully (no content). |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **users_resend_invitation**
> users_resend_invitation(user_id)

Resend a user invitation

Resend the invitation email for a pending (unverified) user. Generates a
new verification token and sends a fresh email to the user's address.

The target user must be in the `invited` state (unverified and active).
Returns 400 if the user has already verified their account, or if
SAML/IdP login is enforced for the account.

This is a fire-and-forget operation: a 204 response means the token was
regenerated and the email dispatch was accepted. If the email fails to send,
the endpoint still returns 204 and logs the error internally.

Requires account admin role or USER_CREATE permission.

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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user

    try:
        # Resend a user invitation
        api_instance.users_resend_invitation(user_id)
    except Exception as e:
        print("Exception when calling UsersApi->users_resend_invitation: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **user_id** | **str**| The unique identifier of the user | 

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
**204** | Invitation resend accepted (no content) |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **users_update**
> User users_update(user_id, user_update)

Update a user

Update a user's display name and/or developer permission.

**Payload Requirements**
- At least one of `name` or `is_developer` must be provided.
- `name` must be 1–255 characters. Leading and trailing whitespace is stripped before
  validation; whitespace-only values (e.g. `"   "`) are rejected with 400.
- Setting `is_developer` to its current value is a no-op (idempotent).

**Example valid requests:**
```json
{ "name": "Jane Smith" }
{ "is_developer": true }
{ "name": "Jane Smith", "is_developer": false }
```

**Example invalid requests:**
- `{}` — at least one field must be provided
- `{ "name": "   " }` — name cannot be whitespace only

Requires account admin role or USER_UPDATE permission.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.user import User
from arize._generated.api_client.models.user_update import UserUpdate
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
    api_instance = arize._generated.api_client.UsersApi(api_client)
    user_id = 'VXNlcjoxMjM0NQ==' # str | The unique identifier of the user
    user_update = {"name":"Jane Smith","is_developer":false} # UserUpdate | Body containing user update parameters. At least one field must be provided.

    try:
        # Update a user
        api_response = api_instance.users_update(user_id, user_update)
        print("The response of UsersApi->users_update:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling UsersApi->users_update: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **user_id** | **str**| The unique identifier of the user | 
 **user_update** | [**UserUpdate**](UserUpdate.md)| Body containing user update parameters. At least one field must be provided. | 

### Return type

[**User**](User.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | An account user object |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**404** | Not found |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

