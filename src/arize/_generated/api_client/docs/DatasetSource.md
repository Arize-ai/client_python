# DatasetSource

Dataset, version, experiment, and fields used as a widget data source.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_id** | **str** | Dataset supplying the values, or null when it is inferred from the experiment. | 
**dataset_version_id** | **str** | Dataset version supplying the values, or null when no version is selected. | 
**experiment_id** | **str** | Experiment supplying evaluation values, or null when no experiment is selected. | 
**source_fields** | [**List[WidgetSourceField]**](WidgetSourceField.md) | One or two dataset fields used to compute the widget value. | 

## Example

```python
from arize._generated.api_client.models.dataset_source import DatasetSource

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetSource from a JSON string
dataset_source_instance = DatasetSource.from_json(json)
# print the JSON string representation of the object
print(DatasetSource.to_json())

# convert the object into a dict
dataset_source_dict = dataset_source_instance.to_dict()
# create an instance of DatasetSource from a dict
dataset_source_from_dict = DatasetSource.from_dict(dataset_source_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


