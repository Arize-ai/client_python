# ExperimentsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the experiment | 
**dataset_id** | **str** | ID of the dataset to create the experiment for | 
**experiment_runs** | [**List[ExperimentRunCreate]**](ExperimentRunCreate.md) | Array of experiment run data | 

## Example

```python
from arize._generated.api_client.models.experiments_create_request import ExperimentsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of ExperimentsCreateRequest from a JSON string
experiments_create_request_instance = ExperimentsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(ExperimentsCreateRequest.to_json())

# convert the object into a dict
experiments_create_request_dict = experiments_create_request_instance.to_dict()
# create an instance of ExperimentsCreateRequest from a dict
experiments_create_request_from_dict = ExperimentsCreateRequest.from_dict(experiments_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


