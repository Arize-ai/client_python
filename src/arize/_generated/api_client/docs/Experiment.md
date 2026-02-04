# Experiment

Experiments combine a dataset (example inputs/expected outputs), a task (the function that produces model outputs), and one or more evaluators (code or LLM judges) to measure performance. Each run is stored independently so you can compare runs, track progress, and validate improvements over time. See the full definition on the Experiments page.  Use an experiment to run tasks on a dataset, attach evaluators to score outputs, and compare runs to confirm improvements. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the experiment | 
**name** | **str** | Name of the experiment | 
**dataset_id** | **str** | Unique identifier for the dataset this experiment belongs to | 
**dataset_version_id** | **str** | Unique identifier for the dataset version this experiment belongs to | 
**created_at** | **datetime** | Timestamp for when the experiment was created | 
**updated_at** | **datetime** | Timestamp for the last update of the experiment | 
**experiment_traces_project_id** | **str** | Unique identifier for the experiment traces project this experiment belongs to (if it exists) | [optional] 

## Example

```python
from arize._generated.api_client.models.experiment import Experiment

# TODO update the JSON string below
json = "{}"
# create an instance of Experiment from a JSON string
experiment_instance = Experiment.from_json(json)
# print the JSON string representation of the object
print(Experiment.to_json())

# convert the object into a dict
experiment_dict = experiment_instance.to_dict()
# create an instance of Experiment from a dict
experiment_from_dict = Experiment.from_dict(experiment_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


