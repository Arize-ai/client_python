# ExperimentRunsListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_runs** | [**List[ExperimentRun]**](ExperimentRun.md) | Array of experiment run objects containing experiment fields and evaluations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.experiment_runs_list_response import ExperimentRunsListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentRunsListResponse from a JSON string
experiment_runs_list_response_instance = ExperimentRunsListResponse.from_json(json)
# print the JSON string representation of the object
print(ExperimentRunsListResponse.to_json())

# convert the object into a dict
experiment_runs_list_response_dict = experiment_runs_list_response_instance.to_dict()
# create an instance of ExperimentRunsListResponse from a dict
experiment_runs_list_response_from_dict = ExperimentRunsListResponse.from_dict(experiment_runs_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


