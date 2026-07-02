# AuditLogsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**logs** | [**List[AuditLog]**](AuditLog.md) | A list of audit log entries, newest first. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.audit_logs_list200_response import AuditLogsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of AuditLogsList200Response from a JSON string
audit_logs_list200_response_instance = AuditLogsList200Response.from_json(json)
# print the JSON string representation of the object
print(AuditLogsList200Response.to_json())

# convert the object into a dict
audit_logs_list200_response_dict = audit_logs_list200_response_instance.to_dict()
# create an instance of AuditLogsList200Response from a dict
audit_logs_list200_response_from_dict = AuditLogsList200Response.from_dict(audit_logs_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


