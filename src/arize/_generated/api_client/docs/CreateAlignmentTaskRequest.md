# CreateAlignmentTaskRequest

Parameters for creating an alignment task.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**dataset_id** | **str** | The identifier of the dataset that contains the labels and annotations | 
**name** | **str** | The name of the alignment task | 
**judge_label_column** | **str** | The dataset column holding the evaluator&#39;s labels, for example &#x60;eval.hallucination.label&#x60; when the labels are already on the dataset, or &#x60;output.label&#x60; when an experiment produces them.  | 
**annotation_column** | **str** | The dataset column holding the human annotations to compare against | 
**alignment_function** | [**AlignmentFunction**](AlignmentFunction.md) | The comparison function. Defaults to &#x60;EXACT_MATCH&#x60; if omitted.  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_alignment_task_request import CreateAlignmentTaskRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateAlignmentTaskRequest from a JSON string
create_alignment_task_request_instance = CreateAlignmentTaskRequest.from_json(json)
# print the JSON string representation of the object
print(CreateAlignmentTaskRequest.to_json())

# convert the object into a dict
create_alignment_task_request_dict = create_alignment_task_request_instance.to_dict()
# create an instance of CreateAlignmentTaskRequest from a dict
create_alignment_task_request_from_dict = CreateAlignmentTaskRequest.from_dict(create_alignment_task_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


