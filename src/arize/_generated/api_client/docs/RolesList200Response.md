# RolesList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**roles** | [**List[Role]**](Role.md) | A list of roles. | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.roles_list200_response import RolesList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of RolesList200Response from a JSON string
roles_list200_response_instance = RolesList200Response.from_json(json)
# print the JSON string representation of the object
print(RolesList200Response.to_json())

# convert the object into a dict
roles_list200_response_dict = roles_list200_response_instance.to_dict()
# create an instance of RolesList200Response from a dict
roles_list200_response_from_dict = RolesList200Response.from_dict(roles_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


