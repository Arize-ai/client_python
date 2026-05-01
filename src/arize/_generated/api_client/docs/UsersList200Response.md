# UsersList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**users** | [**List[User]**](User.md) | A list of account users | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.users_list200_response import UsersList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of UsersList200Response from a JSON string
users_list200_response_instance = UsersList200Response.from_json(json)
# print the JSON string representation of the object
print(UsersList200Response.to_json())

# convert the object into a dict
users_list200_response_dict = users_list200_response_instance.to_dict()
# create an instance of UsersList200Response from a dict
users_list200_response_from_dict = UsersList200Response.from_dict(users_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


