# ResourceRestrictionCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_id** | **str** | The ID of the resource to restrict | 

## Example

```python
from arize._generated.api_client.models.resource_restriction_create import ResourceRestrictionCreate

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestrictionCreate from a JSON string
resource_restriction_create_instance = ResourceRestrictionCreate.from_json(json)
# print the JSON string representation of the object
print(ResourceRestrictionCreate.to_json())

# convert the object into a dict
resource_restriction_create_dict = resource_restriction_create_instance.to_dict()
# create an instance of ResourceRestrictionCreate from a dict
resource_restriction_create_from_dict = ResourceRestrictionCreate.from_dict(resource_restriction_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


