# ExperimentRunCreate

An experiment run with experiment data including outputs, evaluations, and trace metadata

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**example_id** | **str** | ID of the dataset example associated with this experiment run | 
**output** | **str** | output of the task for the matching example | 

## Example

```python
from arize._generated.api_client.models.experiment_run_create import ExperimentRunCreate

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentRunCreate from a JSON string
experiment_run_create_instance = ExperimentRunCreate.from_json(json)
# print the JSON string representation of the object
print(ExperimentRunCreate.to_json())

# convert the object into a dict
experiment_run_create_dict = experiment_run_create_instance.to_dict()
# create an instance of ExperimentRunCreate from a dict
experiment_run_create_from_dict = ExperimentRunCreate.from_dict(experiment_run_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


