# AnnotateExperimentRunsRequest

Batch annotation request for experiment runs.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotations** | [**List[AnnotateRecordInput]**](AnnotateRecordInput.md) | Batch of experiment run annotations to write. Up to 1000 runs per request. | 

## Example

```python
from arize._generated.api_client.models.annotate_experiment_runs_request import AnnotateExperimentRunsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateExperimentRunsRequest from a JSON string
annotate_experiment_runs_request_instance = AnnotateExperimentRunsRequest.from_json(json)
# print the JSON string representation of the object
print(AnnotateExperimentRunsRequest.to_json())

# convert the object into a dict
annotate_experiment_runs_request_dict = annotate_experiment_runs_request_instance.to_dict()
# create an instance of AnnotateExperimentRunsRequest from a dict
annotate_experiment_runs_request_from_dict = AnnotateExperimentRunsRequest.from_dict(annotate_experiment_runs_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


