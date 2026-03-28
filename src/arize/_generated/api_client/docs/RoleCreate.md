# RoleCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Human-readable name for the role. Must be unique within the account. | 
**description** | **str** | Optional description of the role&#39;s purpose. Omitted from the response if empty. | [optional] 
**permissions** | [**List[Permission]**](Permission.md) | List of permissions to grant. At least one permission is required. Each value must be a valid permission identifier (e.g. &#x60;PROJECT_READ&#x60;, &#x60;DATASET_CREATE&#x60;).  | 

## Example

```python
from arize._generated.api_client.models.role_create import RoleCreate

# TODO update the JSON string below
json = "{}"
# create an instance of RoleCreate from a JSON string
role_create_instance = RoleCreate.from_json(json)
# print the JSON string representation of the object
print(RoleCreate.to_json())

# convert the object into a dict
role_create_dict = role_create_instance.to_dict()
# create an instance of RoleCreate from a dict
role_create_from_dict = RoleCreate.from_dict(role_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


