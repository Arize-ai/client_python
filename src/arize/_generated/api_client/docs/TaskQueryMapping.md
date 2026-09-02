# TaskQueryMapping

Maps one evaluator variable to one or more query ids and an attribute path (trace/session shape). 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**variable_name** | **str** | The evaluator template variable this mapping populates. | 
**query_ids** | **List[str]** | Declared query ids (&#x60;A&#x60;-&#x60;E&#x60;) whose matching units feed this variable. An empty list means \&quot;any declared query\&quot; (valid for session-level variables that match all spans in the conversation). Every id must be declared in the task&#39;s &#x60;query_filters.filters&#x60;.  | 
**attribute_path** | **str** | Span attribute path (e.g. &#x60;attributes.input.value&#x60;) resolved within each admitted unit to populate &#x60;variable_name&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.task_query_mapping import TaskQueryMapping

# TODO update the JSON string below
json = "{}"
# create an instance of TaskQueryMapping from a JSON string
task_query_mapping_instance = TaskQueryMapping.from_json(json)
# print the JSON string representation of the object
print(TaskQueryMapping.to_json())

# convert the object into a dict
task_query_mapping_dict = task_query_mapping_instance.to_dict()
# create an instance of TaskQueryMapping from a dict
task_query_mapping_from_dict = TaskQueryMapping.from_dict(task_query_mapping_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


