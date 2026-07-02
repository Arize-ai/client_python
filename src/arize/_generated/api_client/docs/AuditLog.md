# AuditLog

A single audit log entry recording an authenticated user action.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**user_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 
**ip** | **str** | The IP address from which the request originated. | 
**operation_type** | [**AuditLogOperationType**](AuditLogOperationType.md) |  | 
**operation_name** | **str** | The name of the GraphQL operation or REST endpoint. | [optional] 
**operation_text** | **str** | The full text of the operation (query or mutation body, or REST request body). | [optional] 
**variables** | **str** | JSON-serialized variables passed with the operation. | [optional] 
**created_at** | **datetime** | ISO 8601 timestamp when the action was recorded. | 

## Example

```python
from arize._generated.api_client.models.audit_log import AuditLog

# TODO update the JSON string below
json = "{}"
# create an instance of AuditLog from a JSON string
audit_log_instance = AuditLog.from_json(json)
# print the JSON string representation of the object
print(AuditLog.to_json())

# convert the object into a dict
audit_log_dict = audit_log_instance.to_dict()
# create an instance of AuditLog from a dict
audit_log_from_dict = AuditLog.from_dict(audit_log_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


