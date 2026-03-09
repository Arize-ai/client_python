# SpacesUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Updated name for the space (must be unique within the organization) | [optional] 
**description** | **str** | Updated description for the space. Set to an empty string to clear it. | [optional] 

## Example

```python
from arize._generated.api_client.models.spaces_update_request import SpacesUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SpacesUpdateRequest from a JSON string
spaces_update_request_instance = SpacesUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(SpacesUpdateRequest.to_json())

# convert the object into a dict
spaces_update_request_dict = spaces_update_request_instance.to_dict()
# create an instance of SpacesUpdateRequest from a dict
spaces_update_request_from_dict = SpacesUpdateRequest.from_dict(spaces_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


