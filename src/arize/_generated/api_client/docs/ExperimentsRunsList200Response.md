# ExperimentsRunsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiment_runs** | [**List[ExperimentRun]**](ExperimentRun.md) | Array of experiment run objects containing experiment fields and evaluations | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.experiments_runs_list200_response import ExperimentsRunsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentsRunsList200Response from a JSON string
experiments_runs_list200_response_instance = ExperimentsRunsList200Response.from_json(json)
# print the JSON string representation of the object
print(ExperimentsRunsList200Response.to_json())

# convert the object into a dict
experiments_runs_list200_response_dict = experiments_runs_list200_response_instance.to_dict()
# create an instance of ExperimentsRunsList200Response from a dict
experiments_runs_list200_response_from_dict = ExperimentsRunsList200Response.from_dict(experiments_runs_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


