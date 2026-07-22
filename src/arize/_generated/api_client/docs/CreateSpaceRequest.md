# CreateSpaceRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the space | 
**organization_id** | **str** | The unique identifier of the organization to create the space in | 
**description** | **str** | A brief description of the space&#39;s purpose | [optional] 

## Example

```python
from arize._generated.api_client.models.create_space_request import CreateSpaceRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateSpaceRequest from a JSON string
create_space_request_instance = CreateSpaceRequest.from_json(json)
# print the JSON string representation of the object
print(CreateSpaceRequest.to_json())

# convert the object into a dict
create_space_request_dict = create_space_request_instance.to_dict()
# create an instance of CreateSpaceRequest from a dict
create_space_request_from_dict = CreateSpaceRequest.from_dict(create_space_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


