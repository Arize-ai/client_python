# ListUsersResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**users** | [**List[User]**](User.md) | A list of account users | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_users_response import ListUsersResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListUsersResponse from a JSON string
list_users_response_instance = ListUsersResponse.from_json(json)
# print the JSON string representation of the object
print(ListUsersResponse.to_json())

# convert the object into a dict
list_users_response_dict = list_users_response_instance.to_dict()
# create an instance of ListUsersResponse from a dict
list_users_response_from_dict = ListUsersResponse.from_dict(list_users_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


