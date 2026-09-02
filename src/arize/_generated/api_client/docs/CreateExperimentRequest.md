# CreateExperimentRequest

Experiment creation parameters with an initial set of runs.  An experiment belongs to a space and may optionally be associated with a dataset. Provide exactly one of: - `dataset_id` — associate the experiment with a dataset; it's created in   that dataset's space, and its runs may reference the dataset's examples   via `example_id`. - `space_id` — the space to create the experiment in, when it isn't   associated with a dataset.  Providing both, or neither, is a validation error. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the experiment. Must be 1–255 characters and must not contain double quotes (&#x60;\&quot;&#x60;) or backslashes (&#x60;\\&#x60;).  | 
**dataset_id** | **str** | ID of the dataset to associate the experiment with. Provide &#x60;space_id&#x60; instead when the experiment isn&#39;t associated with a dataset. | [optional] 
**space_id** | **str** | ID of the space to create the experiment in. Provide instead of &#x60;dataset_id&#x60;. | [optional] 
**experiment_runs** | [**List[ExperimentRunInput]**](ExperimentRunInput.md) | Array of experiment run data. Between 1 and 1000 runs per request. | 

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


