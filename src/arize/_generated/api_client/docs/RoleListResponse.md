# RoleListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**roles** | [**List[Role]**](Role.md) | A list of roles. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.role_list_response import RoleListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of RoleListResponse from a JSON string
role_list_response_instance = RoleListResponse.from_json(json)
# print the JSON string representation of the object
print(RoleListResponse.to_json())

# convert the object into a dict
role_list_response_dict = role_list_response_instance.to_dict()
# create an instance of RoleListResponse from a dict
role_list_response_from_dict = RoleListResponse.from_dict(role_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


