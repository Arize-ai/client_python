# Dataset

A dataset is a structured collection of examples used to test and evaluate LLM applications. Datasets allow you to test models consistently across any real-world scenarios and edge cases, quickly identify regressions, and track measurable improvements. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the dataset | 
**name** | **str** | Name of the dataset | 
**space_id** | **str** | Unique identifier for the space this dataset belongs to | 
**created_at** | **datetime** | Timestamp for when the dataset was created | 
**updated_at** | **datetime** | Timestamp for the last update of the dataset | 
**versions** | [**List[DatasetVersion]**](DatasetVersion.md) | List of versions associated with this dataset | [optional] 

## Example

```python
from arize._generated.api_client.models.dataset import Dataset

# TODO update the JSON string below
json = "{}"
# create an instance of Dataset from a JSON string
dataset_instance = Dataset.from_json(json)
# print the JSON string representation of the object
print(Dataset.to_json())

# convert the object into a dict
dataset_dict = dataset_instance.to_dict()
# create an instance of Dataset from a dict
dataset_from_dict = Dataset.from_dict(dataset_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


