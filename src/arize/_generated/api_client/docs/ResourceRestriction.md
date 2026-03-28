# ResourceRestriction


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_type** | **str** | The type of the restricted resource (e.g. \&quot;PROJECT\&quot;) | 
**resource_id** | **str** | The ID of the restricted resource | 
**created_at** | **datetime** | When the restriction was created | 

## Example

```python
from arize._generated.api_client.models.resource_restriction import ResourceRestriction

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestriction from a JSON string
resource_restriction_instance = ResourceRestriction.from_json(json)
# print the JSON string representation of the object
print(ResourceRestriction.to_json())

# convert the object into a dict
resource_restriction_dict = resource_restriction_instance.to_dict()
# create an instance of ResourceRestriction from a dict
resource_restriction_from_dict = ResourceRestriction.from_dict(resource_restriction_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


