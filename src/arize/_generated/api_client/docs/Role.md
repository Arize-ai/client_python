# Role


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the role. | 
**name** | **str** | Human-readable name of the role. | 
**description** | **str** | A brief description of the role&#39;s purpose. | [optional] 
**permissions** | [**List[Permission]**](Permission.md) | List of permissions granted by this role. Each value corresponds to a permission identifier (e.g. &#x60;PROJECT_READ&#x60;, &#x60;DATASET_CREATE&#x60;).  | 
**is_predefined** | **bool** | Whether this role is a system-defined predefined role. Predefined roles cannot be updated or deleted.  | 
**created_at** | **datetime** | Timestamp when the role was created. | 
**updated_at** | **datetime** | Timestamp when the role was last updated. | 

## Example

```python
from arize._generated.api_client.models.role import Role

# TODO update the JSON string below
json = "{}"
# create an instance of Role from a JSON string
role_instance = Role.from_json(json)
# print the JSON string representation of the object
print(Role.to_json())

# convert the object into a dict
role_dict = role_instance.to_dict()
# create an instance of Role from a dict
role_from_dict = Role.from_dict(role_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


