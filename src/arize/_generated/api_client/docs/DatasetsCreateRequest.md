# DatasetsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new dataset | 
**space_id** | **str** | ID of the space the dataset will belong to | 
**examples** | **List[Dict[str, object]]** | Array of examples for the new dataset | 

## Example

```python
from arize._generated.api_client.models.datasets_create_request import DatasetsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetsCreateRequest from a JSON string
datasets_create_request_instance = DatasetsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(DatasetsCreateRequest.to_json())

# convert the object into a dict
datasets_create_request_dict = datasets_create_request_instance.to_dict()
# create an instance of DatasetsCreateRequest from a dict
datasets_create_request_from_dict = DatasetsCreateRequest.from_dict(datasets_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


