# CreateExperimentRequest

Experiment creation parameters with an initial set of runs.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the experiment | 
**dataset_id** | **str** | ID of the dataset to create the experiment for | 
**experiment_runs** | [**List[ExperimentRunInput]**](ExperimentRunInput.md) | Array of experiment run data | 

## Example

```python
from arize._generated.api_client.models.create_experiment_request import CreateExperimentRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateExperimentRequest from a JSON string
create_experiment_request_instance = CreateExperimentRequest.from_json(json)
# print the JSON string representation of the object
print(CreateExperimentRequest.to_json())

# convert the object into a dict
create_experiment_request_dict = create_experiment_request_instance.to_dict()
# create an instance of CreateExperimentRequest from a dict
create_experiment_request_from_dict = CreateExperimentRequest.from_dict(create_experiment_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


