# SpacesCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the space (must be unique within the organization) | 
**organization_id** | **str** | ID of the organization to create the space in | 
**description** | **str** | A brief description of the space&#39;s purpose. Defaults to an empty string if omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.spaces_create_request import SpacesCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SpacesCreateRequest from a JSON string
spaces_create_request_instance = SpacesCreateRequest.from_json(json)
# print the JSON string representation of the object
print(SpacesCreateRequest.to_json())

# convert the object into a dict
spaces_create_request_dict = spaces_create_request_instance.to_dict()
# create an instance of SpacesCreateRequest from a dict
spaces_create_request_from_dict = SpacesCreateRequest.from_dict(spaces_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


