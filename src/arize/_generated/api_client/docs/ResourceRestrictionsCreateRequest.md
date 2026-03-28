# ResourceRestrictionsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_id** | **str** | The ID of the resource to restrict | 

## Example

```python
from arize._generated.api_client.models.resource_restrictions_create_request import ResourceRestrictionsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ResourceRestrictionsCreateRequest from a JSON string
resource_restrictions_create_request_instance = ResourceRestrictionsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(ResourceRestrictionsCreateRequest.to_json())

# convert the object into a dict
resource_restrictions_create_request_dict = resource_restrictions_create_request_instance.to_dict()
# create an instance of ResourceRestrictionsCreateRequest from a dict
resource_restrictions_create_request_from_dict = ResourceRestrictionsCreateRequest.from_dict(resource_restrictions_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


