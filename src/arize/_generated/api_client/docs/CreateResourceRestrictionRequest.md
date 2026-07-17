# CreateResourceRestrictionRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**resource_id** | **str** | The ID of the resource to restrict | 

## Example

```python
from arize._generated.api_client.models.create_resource_restriction_request import CreateResourceRestrictionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateResourceRestrictionRequest from a JSON string
create_resource_restriction_request_instance = CreateResourceRestrictionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateResourceRestrictionRequest.to_json())

# convert the object into a dict
create_resource_restriction_request_dict = create_resource_restriction_request_instance.to_dict()
# create an instance of CreateResourceRestrictionRequest from a dict
create_resource_restriction_request_from_dict = CreateResourceRestrictionRequest.from_dict(create_resource_restriction_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


