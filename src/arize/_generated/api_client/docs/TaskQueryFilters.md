# TaskQueryFilters

Combined named-query filters and boolean expression for the trace/session shape. Supply this object OR `query_filter` (span shape) — not both. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**filters** | [**List[TaskQueryFilter]**](TaskQueryFilter.md) | Named query filters (1-5 entries) with unique &#x60;A&#x60;-&#x60;E&#x60; ids. Each entry pairs a single-letter id with a filter expression.  | 
**expression** | **str** | Boolean expression combining the &#x60;filters&#x60; ids (e.g. &#x60;A AND B&#x60;). Optional when exactly one filter is declared; required when two or more are declared. When a client omits &#x60;expression&#x60; on a single-filter create, GET echoes a synthesized expression equal to the lone query id (e.g. &#x60;A&#x60;).  | [optional] 

## Example

```python
from arize._generated.api_client.models.task_query_filters import TaskQueryFilters

# TODO update the JSON string below
json = "{}"
# create an instance of TaskQueryFilters from a JSON string
task_query_filters_instance = TaskQueryFilters.from_json(json)
# print the JSON string representation of the object
print(TaskQueryFilters.to_json())

# convert the object into a dict
task_query_filters_dict = task_query_filters_instance.to_dict()
# create an instance of TaskQueryFilters from a dict
task_query_filters_from_dict = TaskQueryFilters.from_dict(task_query_filters_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


