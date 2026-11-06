# CreateAlignmentRunRequest

Body containing alignment run parameters.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_version_id** | **str** | The unique identifier for the evaluator version to score. The caller must have update access to the evaluator that owns it.  | 
**dataset_version_id** | **str** | The unique identifier for the dataset version to score over. Must belong to the alignment task&#39;s dataset.  | 
**predicted_entity** | [**AlignmentPredictedEntity**](AlignmentPredictedEntity.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_alignment_run_request import CreateAlignmentRunRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAlignmentRunRequest from a JSON string
create_alignment_run_request_instance = CreateAlignmentRunRequest.from_json(json)
# print the JSON string representation of the object
print(CreateAlignmentRunRequest.to_json())

# convert the object into a dict
create_alignment_run_request_dict = create_alignment_run_request_instance.to_dict()
# create an instance of CreateAlignmentRunRequest from a dict
create_alignment_run_request_from_dict = CreateAlignmentRunRequest.from_dict(create_alignment_run_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


