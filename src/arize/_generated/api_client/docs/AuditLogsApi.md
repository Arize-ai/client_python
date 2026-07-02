# arize._generated.api_client.AuditLogsApi

All URIs are relative to *https://api.arize.com*

Method | HTTP request | Description
------------- | ------------- | -------------
[**audit_logs_list**](AuditLogsApi.md#audit_logs_list) | **GET** /v2/audit-logs | List audit logs


# **audit_logs_list**
> AuditLogsList200Response audit_logs_list(start_time=start_time, end_time=end_time, user_id=user_id, operation_type=operation_type, limit=limit, cursor=cursor)

List audit logs

Retrieve a paginated list of authenticated user audit log entries for the
account. Results are ordered newest first.

**Access requirements:**
- The caller must be an account admin.
- The account must have audit logging enabled.

Returns `403` if either condition is not met.

<Warning>This endpoint is in alpha, read more [here](https://arize.com/docs/ax/rest-reference#api-version-stages).</Warning>


### Example

* Bearer (<api-key>) Authentication (bearerAuth):

```python
import arize._generated.api_client
from arize._generated.api_client.models.audit_log_operation_type import AuditLogOperationType
from arize._generated.api_client.models.audit_logs_list200_response import AuditLogsList200Response
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
    api_instance = arize._generated.api_client.AuditLogsApi(api_client)
    start_time = '2026-04-18T00:00:00Z' # datetime | Inclusive lower bound on `created_at` (ISO 8601 datetime). Defaults to 30 days before `end_time` when omitted.  (optional)
    end_time = '2026-05-18T23:59:59Z' # datetime | Inclusive upper bound on `created_at` (ISO 8601 datetime). Defaults to the current time when omitted.  (optional)
    user_id = 'VXNlcjoxMjM0NQ==' # str | Filter results by user (base64 global user ID). When provided, only records associated with this user are returned. Access requirements vary by endpoint — some endpoints restrict this filter to account admins.  (optional)
    operation_type = arize._generated.api_client.AuditLogOperationType() # AuditLogOperationType | Filter results to a specific operation type. (optional)
    limit = 50 # int | Maximum items to return (optional) (default to 50)
    cursor = 'cursor_example' # str | Opaque pagination cursor returned from a previous response (`pagination.next_cursor`). Treat it as an unreadable token; do not attempt to parse or construct it.  (optional)

    try:
        # List audit logs
        api_response = api_instance.audit_logs_list(start_time=start_time, end_time=end_time, user_id=user_id, operation_type=operation_type, limit=limit, cursor=cursor)
        print("The response of AuditLogsApi->audit_logs_list:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling AuditLogsApi->audit_logs_list: %s\n" % e)
```



### Parameters


Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **start_time** | **datetime**| Inclusive lower bound on &#x60;created_at&#x60; (ISO 8601 datetime). Defaults to 30 days before &#x60;end_time&#x60; when omitted.  | [optional] 
 **end_time** | **datetime**| Inclusive upper bound on &#x60;created_at&#x60; (ISO 8601 datetime). Defaults to the current time when omitted.  | [optional] 
 **user_id** | **str**| Filter results by user (base64 global user ID). When provided, only records associated with this user are returned. Access requirements vary by endpoint — some endpoints restrict this filter to account admins.  | [optional] 
 **operation_type** | [**AuditLogOperationType**](.md)| Filter results to a specific operation type. | [optional] 
 **limit** | **int**| Maximum items to return | [optional] [default to 50]
 **cursor** | **str**| Opaque pagination cursor returned from a previous response (&#x60;pagination.next_cursor&#x60;). Treat it as an unreadable token; do not attempt to parse or construct it.  | [optional] 

### Return type

[**AuditLogsList200Response**](AuditLogsList200Response.md)

### Authorization

[bearerAuth](../README.md#bearerAuth)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/problem+json

### HTTP response details

| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | A paginated list of audit log entries. |  -  |
**400** | Invalid request |  -  |
**401** | Authentication is required |  -  |
**403** | Insufficient permissions to access this resource |  -  |
**429** | Rate limit exceeded |  * Retry-After - When throttled (429), how long to wait before retrying. Value is either a delta-seconds integer.  <br>  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

