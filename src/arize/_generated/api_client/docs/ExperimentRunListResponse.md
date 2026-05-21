# ExperimentRunListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_runs** | [**List[ExperimentRun]**](ExperimentRun.md) | Array of experiment run objects containing experiment fields and evaluations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.experiment_run_list_response import ExperimentRunListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentRunListResponse from a JSON string
experiment_run_list_response_instance = ExperimentRunListResponse.from_json(json)
# print the JSON string representation of the object
print(ExperimentRunListResponse.to_json())

# convert the object into a dict
experiment_run_list_response_dict = experiment_run_list_response_instance.to_dict()
# create an instance of ExperimentRunListResponse from a dict
experiment_run_list_response_from_dict = ExperimentRunListResponse.from_dict(experiment_run_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


