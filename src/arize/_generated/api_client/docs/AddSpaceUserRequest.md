# AddSpaceUserRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**user_id** | **str** | The unique identifier of the user to add | 
**role** | [**SpaceRoleAssignment**](SpaceRoleAssignment.md) |  | 

## Example

```python
from arize._generated.api_client.models.add_space_user_request import AddSpaceUserRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AddSpaceUserRequest from a JSON string
add_space_user_request_instance = AddSpaceUserRequest.from_json(json)
# print the JSON string representation of the object
print(AddSpaceUserRequest.to_json())

# convert the object into a dict
add_space_user_request_dict = add_space_user_request_instance.to_dict()
# create an instance of AddSpaceUserRequest from a dict
add_space_user_request_from_dict = AddSpaceUserRequest.from_dict(add_space_user_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


