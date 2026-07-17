# ListAuditLogsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**logs** | [**List[AuditLog]**](AuditLog.md) | A list of audit log entries, newest first. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_audit_logs_response import ListAuditLogsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListAuditLogsResponse from a JSON string
list_audit_logs_response_instance = ListAuditLogsResponse.from_json(json)
# print the JSON string representation of the object
print(ListAuditLogsResponse.to_json())

# convert the object into a dict
list_audit_logs_response_dict = list_audit_logs_response_instance.to_dict()
# create an instance of ListAuditLogsResponse from a dict
list_audit_logs_response_from_dict = ListAuditLogsResponse.from_dict(list_audit_logs_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


