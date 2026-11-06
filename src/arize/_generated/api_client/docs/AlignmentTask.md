# AlignmentTask

A reusable comparison of evaluator labels and human annotations on a dataset. The dataset, label column, annotation column, and comparison function identify the task. The name does not affect task identity. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the alignment task | 
**created_at** | **datetime** | When the alignment task was created | 
**updated_at** | **datetime** | When the alignment task was last modified | 
**name** | **str** | The name of the alignment task | 
**dataset_id** | **str** | The identifier of the dataset that contains the labels and annotations | 
**judge_label_column** | **str** | The dataset column holding the evaluator&#39;s labels, for example &#x60;eval.hallucination.label&#x60; when the labels are already on the dataset, or &#x60;output.label&#x60; when an experiment produces them.  | 
**annotation_column** | **str** | The dataset column holding the human annotations to compare against | 
**alignment_function** | [**AlignmentFunction**](AlignmentFunction.md) |  | 

## Example

```python
from arize._generated.api_client.models.alignment_task import AlignmentTask

# TODO update the JSON string below
json = "{}"
# create an instance of AlignmentTask from a JSON string
alignment_task_instance = AlignmentTask.from_json(json)
# print the JSON string representation of the object
print(AlignmentTask.to_json())

# convert the object into a dict
alignment_task_dict = alignment_task_instance.to_dict()
# create an instance of AlignmentTask from a dict
alignment_task_from_dict = AlignmentTask.from_dict(alignment_task_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


