# ResourceRestrictionCreateResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_restriction** | [**ResourceRestriction**](ResourceRestriction.md) |  | 

## Example

```python
from arize._generated.api_client.models.resource_restriction_create_response import ResourceRestrictionCreateResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestrictionCreateResponse from a JSON string
resource_restriction_create_response_instance = ResourceRestrictionCreateResponse.from_json(json)
# print the JSON string representation of the object
print(ResourceRestrictionCreateResponse.to_json())

# convert the object into a dict
resource_restriction_create_response_dict = resource_restriction_create_response_instance.to_dict()
# create an instance of ResourceRestrictionCreateResponse from a dict
resource_restriction_create_response_from_dict = ResourceRestrictionCreateResponse.from_dict(resource_restriction_create_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


