# DatasetVersionWithExampleIds

A dataset with the IDs of examples that were inserted or updated. Includes the version the examples were written to and the list of affected example IDs. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the dataset | 
**name** | **str** | Name of the dataset | 
**space_id** | **str** | Unique identifier for the space this dataset belongs to | 
**created_at** | **datetime** | Timestamp for when the dataset was created | 
**updated_at** | **datetime** | Timestamp for the last update of the dataset | 
**dataset_version_id** | **str** | Unique identifier for the dataset version the examples were written to | 
**example_ids** | **List[str]** | IDs of the examples that were inserted or updated | 

## Example

```python
from arize._generated.api_client.models.dataset_version_with_example_ids import DatasetVersionWithExampleIds

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetVersionWithExampleIds from a JSON string
dataset_version_with_example_ids_instance = DatasetVersionWithExampleIds.from_json(json)
# print the JSON string representation of the object
print(DatasetVersionWithExampleIds.to_json())

# convert the object into a dict
dataset_version_with_example_ids_dict = dataset_version_with_example_ids_instance.to_dict()
# create an instance of DatasetVersionWithExampleIds from a dict
dataset_version_with_example_ids_from_dict = DatasetVersionWithExampleIds.from_dict(dataset_version_with_example_ids_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


