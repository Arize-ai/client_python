# ResourceRestrictionsCreate200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_restriction** | [**ResourceRestriction**](ResourceRestriction.md) |  | 

## Example

```python
from arize._generated.api_client.models.resource_restrictions_create200_response import ResourceRestrictionsCreate200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestrictionsCreate200Response from a JSON string
resource_restrictions_create200_response_instance = ResourceRestrictionsCreate200Response.from_json(json)
# print the JSON string representation of the object
print(ResourceRestrictionsCreate200Response.to_json())

# convert the object into a dict
resource_restrictions_create200_response_dict = resource_restrictions_create200_response_instance.to_dict()
# create an instance of ResourceRestrictionsCreate200Response from a dict
resource_restrictions_create200_response_from_dict = ResourceRestrictionsCreate200Response.from_dict(resource_restrictions_create200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


