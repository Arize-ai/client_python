# DatasetVersion

A dataset version is a saved snapshot of a dataset. Arize stores datasets as version-controlled collections of examples for experiments. When you update a dataset, you can do so inplace  or create a new version while preserving access to previous versions. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the dataset version | 
**name** | **str** | Name of the dataset version | 
**dataset_id** | **str** | Unique identifier for the dataset this version belongs to | 
**created_at** | **datetime** | Timestamp for when the dataset version was created | 
**updated_at** | **datetime** | Timestamp for the last update of the dataset version | 

## Example

```python
from arize._generated.api_client.models.dataset_version import DatasetVersion

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetVersion from a JSON string
dataset_version_instance = DatasetVersion.from_json(json)
# print the JSON string representation of the object
print(DatasetVersion.to_json())

# convert the object into a dict
dataset_version_dict = dataset_version_instance.to_dict()
# create an instance of DatasetVersion from a dict
dataset_version_from_dict = DatasetVersion.from_dict(dataset_version_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


