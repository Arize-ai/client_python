# UpdateSpaceRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Updated name for the space (must be unique within the organization) | [optional] 
**description** | **str** | Updated description for the space. Set to an empty string to clear it. | [optional] 

## Example

```python
from arize._generated.api_client.models.update_space_request import UpdateSpaceRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateSpaceRequest from a JSON string
update_space_request_instance = UpdateSpaceRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateSpaceRequest.to_json())

# convert the object into a dict
update_space_request_dict = update_space_request_instance.to_dict()
# create an instance of UpdateSpaceRequest from a dict
update_space_request_from_dict = UpdateSpaceRequest.from_dict(update_space_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


