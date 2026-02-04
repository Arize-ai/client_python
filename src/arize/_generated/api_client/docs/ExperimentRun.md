# ExperimentRun

An experiment run with experiment data including outputs, evaluations, and trace metadata

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | System-assigned unique ID for the example | [readonly] 
**example_id** | **str** | ID of the dataset example associated with this experiment run | [readonly] 
**output** | **str** | output of the task for the matching example | 

## Example

```python
from arize._generated.api_client.models.experiment_run import ExperimentRun

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentRun from a JSON string
experiment_run_instance = ExperimentRun.from_json(json)
# print the JSON string representation of the object
print(ExperimentRun.to_json())

# convert the object into a dict
experiment_run_dict = experiment_run_instance.to_dict()
# create an instance of ExperimentRun from a dict
experiment_run_from_dict = ExperimentRun.from_dict(experiment_run_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


