# InsertExperimentRunsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_runs** | [**List[ExperimentRunInput]**](ExperimentRunInput.md) | Array of experiment run data to append to the experiment. Between 1 and 1000 runs per request. | 

## Example

```python
from arize._generated.api_client.models.insert_experiment_runs_request import InsertExperimentRunsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of InsertExperimentRunsRequest from a JSON string
insert_experiment_runs_request_instance = InsertExperimentRunsRequest.from_json(json)
# print the JSON string representation of the object
print(InsertExperimentRunsRequest.to_json())

# convert the object into a dict
insert_experiment_runs_request_dict = insert_experiment_runs_request_instance.to_dict()
# create an instance of InsertExperimentRunsRequest from a dict
insert_experiment_runs_request_from_dict = InsertExperimentRunsRequest.from_dict(insert_experiment_runs_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


