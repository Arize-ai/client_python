# AddOrganizationUserRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**user_id** | **str** | The unique identifier of the user to add | 
**role** | [**OrganizationRoleAssignment**](OrganizationRoleAssignment.md) |  | 

## Example

```python
from arize._generated.api_client.models.add_organization_user_request import AddOrganizationUserRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AddOrganizationUserRequest from a JSON string
add_organization_user_request_instance = AddOrganizationUserRequest.from_json(json)
# print the JSON string representation of the object
print(AddOrganizationUserRequest.to_json())

# convert the object into a dict
add_organization_user_request_dict = add_organization_user_request_instance.to_dict()
# create an instance of AddOrganizationUserRequest from a dict
add_organization_user_request_from_dict = AddOrganizationUserRequest.from_dict(add_organization_user_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


