# UserListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**users** | [**List[User]**](User.md) | A list of account users | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.user_list_response import UserListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of UserListResponse from a JSON string
user_list_response_instance = UserListResponse.from_json(json)
# print the JSON string representation of the object
print(UserListResponse.to_json())

# convert the object into a dict
user_list_response_dict = user_list_response_instance.to_dict()
# create an instance of UserListResponse from a dict
user_list_response_from_dict = UserListResponse.from_dict(user_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


