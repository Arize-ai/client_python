# TaskQueryFilterInput

A single named task-level query filter for create/update requests (trace/session shape). The `id` is a logical label used in `query_filters.expression` and per-evaluator `query_mappings`. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Single-letter query id, one of &#x60;A&#x60;-&#x60;E&#x60;. Unique within the task. Referenced by &#x60;query_filters.expression&#x60; and by each evaluator&#39;s &#x60;query_mappings&#x60;.  | 
**filter** | **str** | The query filter expression for this named query. | 

## Example

```python
from arize._generated.api_client.models.task_query_filter_input import TaskQueryFilterInput

# TODO update the JSON string below
json = "{}"
# create an instance of TaskQueryFilterInput from a JSON string
task_query_filter_input_instance = TaskQueryFilterInput.from_json(json)
# print the JSON string representation of the object
print(TaskQueryFilterInput.to_json())

# convert the object into a dict
task_query_filter_input_dict = task_query_filter_input_instance.to_dict()
# create an instance of TaskQueryFilterInput from a dict
task_query_filter_input_from_dict = TaskQueryFilterInput.from_dict(task_query_filter_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


