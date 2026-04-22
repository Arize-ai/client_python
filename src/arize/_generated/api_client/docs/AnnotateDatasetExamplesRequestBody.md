# AnnotateDatasetExamplesRequestBody

Batch annotation request for dataset examples.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotations** | [**List[AnnotateRecordInput]**](AnnotateRecordInput.md) | Batch of dataset example annotations to write. Up to 500 examples per request. | 

## Example

```python
from arize._generated.api_client.models.annotate_dataset_examples_request_body import AnnotateDatasetExamplesRequestBody

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateDatasetExamplesRequestBody from a JSON string
annotate_dataset_examples_request_body_instance = AnnotateDatasetExamplesRequestBody.from_json(json)
# print the JSON string representation of the object
print(AnnotateDatasetExamplesRequestBody.to_json())

# convert the object into a dict
annotate_dataset_examples_request_body_dict = annotate_dataset_examples_request_body_instance.to_dict()
# create an instance of AnnotateDatasetExamplesRequestBody from a dict
annotate_dataset_examples_request_body_from_dict = AnnotateDatasetExamplesRequestBody.from_dict(annotate_dataset_examples_request_body_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


