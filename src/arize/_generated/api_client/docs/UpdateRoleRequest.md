# UpdateRoleRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Updated name for the role. Must be unique within the account. | [optional] 
**description** | **str** | Updated description of the role. | [optional] 
**permissions** | [**List[Permission]**](Permission.md) | Replacement set of permissions. When provided, the existing permissions are fully replaced. Each value must be a valid permission identifier.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_role_request import UpdateRoleRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateRoleRequest from a JSON string
update_role_request_instance = UpdateRoleRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateRoleRequest.to_json())

# convert the object into a dict
update_role_request_dict = update_role_request_instance.to_dict()
# create an instance of UpdateRoleRequest from a dict
update_role_request_from_dict = UpdateRoleRequest.from_dict(update_role_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


