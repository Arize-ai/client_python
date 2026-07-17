# UpdateRoleBindingRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**role_id** | **str** | A universally unique identifier (base64-encoded opaque string). | 

## Example

```python
from arize._generated.api_client.models.update_role_binding_request import UpdateRoleBindingRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateRoleBindingRequest from a JSON string
update_role_binding_request_instance = UpdateRoleBindingRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateRoleBindingRequest.to_json())

# convert the object into a dict
update_role_binding_request_dict = update_role_binding_request_instance.to_dict()
# create an instance of UpdateRoleBindingRequest from a dict
update_role_binding_request_from_dict = UpdateRoleBindingRequest.from_dict(update_role_binding_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


