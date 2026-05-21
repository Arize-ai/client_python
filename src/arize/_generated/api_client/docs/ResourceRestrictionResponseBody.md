# ResourceRestrictionResponseBody


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_restriction** | [**ResourceRestriction**](ResourceRestriction.md) |  | 

## Example

```python
from arize._generated.api_client.models.resource_restriction_response_body import ResourceRestrictionResponseBody

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestrictionResponseBody from a JSON string
resource_restriction_response_body_instance = ResourceRestrictionResponseBody.from_json(json)
# print the JSON string representation of the object
print(ResourceRestrictionResponseBody.to_json())

# convert the object into a dict
resource_restriction_response_body_dict = resource_restriction_response_body_instance.to_dict()
# create an instance of ResourceRestrictionResponseBody from a dict
resource_restriction_response_body_from_dict = ResourceRestrictionResponseBody.from_dict(resource_restriction_response_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


