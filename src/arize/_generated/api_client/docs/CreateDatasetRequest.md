# CreateDatasetRequest

Dataset creation parameters.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new dataset. Must be unique within the space. | 
**space_id** | **str** | ID of the space the dataset will belong to | 
**examples** | **List[Dict[str, object]]** | Array of examples for the new dataset | 

## Example

```python
from arize._generated.api_client.models.create_dataset_request import CreateDatasetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDatasetRequest from a JSON string
create_dataset_request_instance = CreateDatasetRequest.from_json(json)
# print the JSON string representation of the object
print(CreateDatasetRequest.to_json())

# convert the object into a dict
create_dataset_request_dict = create_dataset_request_instance.to_dict()
# create an instance of CreateDatasetRequest from a dict
create_dataset_request_from_dict = CreateDatasetRequest.from_dict(create_dataset_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


