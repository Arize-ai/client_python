# ListExperimentRunsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_runs** | [**List[ExperimentRun]**](ExperimentRun.md) | Array of experiment run objects containing experiment fields and evaluations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_experiment_runs_response import ListExperimentRunsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListExperimentRunsResponse from a JSON string
list_experiment_runs_response_instance = ListExperimentRunsResponse.from_json(json)
# print the JSON string representation of the object
print(ListExperimentRunsResponse.to_json())

# convert the object into a dict
list_experiment_runs_response_dict = list_experiment_runs_response_instance.to_dict()
# create an instance of ListExperimentRunsResponse from a dict
list_experiment_runs_response_from_dict = ListExperimentRunsResponse.from_dict(list_experiment_runs_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


