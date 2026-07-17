# ExperimentRunInput

An experiment run with experiment data including outputs, evaluations, and trace metadata

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**example_id** | **str** | ID of the dataset example associated with this experiment run | 
**output** | **str** | output of the task for the matching example | 

## Example

```python
from arize._generated.api_client.models.experiment_run_input import ExperimentRunInput

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentRunInput from a JSON string
experiment_run_input_instance = ExperimentRunInput.from_json(json)
# print the JSON string representation of the object
print(ExperimentRunInput.to_json())

# convert the object into a dict
experiment_run_input_dict = experiment_run_input_instance.to_dict()
# create an instance of ExperimentRunInput from a dict
experiment_run_input_from_dict = ExperimentRunInput.from_dict(experiment_run_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


