# CreateDatasetSource

Dataset, version, experiment, and fields used by a chart plot.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_id** | **str** | ID of the dataset. | 
**dataset_version_id** | **str** | ID of the dataset version. The latest version is used when omitted. | [optional] 
**experiment_id** | **str** | ID of the experiment whose results are displayed. | [optional] 
**source_fields** | [**List[CreateWidgetSourceField]**](CreateWidgetSourceField.md) | Dataset fields displayed by the plot. | 

## Example

```python
from arize._generated.api_client.models.create_dataset_source import CreateDatasetSource

# TODO update the JSON string below
json = "{}"
# create an instance of CreateDatasetSource from a JSON string
create_dataset_source_instance = CreateDatasetSource.from_json(json)
# print the JSON string representation of the object
print(CreateDatasetSource.to_json())

# convert the object into a dict
create_dataset_source_dict = create_dataset_source_instance.to_dict()
# create an instance of CreateDatasetSource from a dict
create_dataset_source_from_dict = CreateDatasetSource.from_dict(create_dataset_source_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


