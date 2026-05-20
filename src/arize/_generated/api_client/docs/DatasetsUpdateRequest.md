# DatasetsUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New name for the dataset. Must be unique within the space. | 

## Example

```python
from arize._generated.api_client.models.datasets_update_request import DatasetsUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetsUpdateRequest from a JSON string
datasets_update_request_instance = DatasetsUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(DatasetsUpdateRequest.to_json())

# convert the object into a dict
datasets_update_request_dict = datasets_update_request_instance.to_dict()
# create an instance of DatasetsUpdateRequest from a dict
datasets_update_request_from_dict = DatasetsUpdateRequest.from_dict(datasets_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


