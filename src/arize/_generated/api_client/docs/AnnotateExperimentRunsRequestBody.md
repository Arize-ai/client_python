# AnnotateExperimentRunsRequestBody

Batch annotation request for experiment runs.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotations** | [**List[AnnotateRecordInput]**](AnnotateRecordInput.md) | Batch of experiment run annotations to write. Up to 500 runs per request. | 

## Example

```python
from arize._generated.api_client.models.annotate_experiment_runs_request_body import AnnotateExperimentRunsRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateExperimentRunsRequestBody from a JSON string
annotate_experiment_runs_request_body_instance = AnnotateExperimentRunsRequestBody.from_json(json)
# print the JSON string representation of the object
print(AnnotateExperimentRunsRequestBody.to_json())

# convert the object into a dict
annotate_experiment_runs_request_body_dict = annotate_experiment_runs_request_body_instance.to_dict()
# create an instance of AnnotateExperimentRunsRequestBody from a dict
annotate_experiment_runs_request_body_from_dict = AnnotateExperimentRunsRequestBody.from_dict(annotate_experiment_runs_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


