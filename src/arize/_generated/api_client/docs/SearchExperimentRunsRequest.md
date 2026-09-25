# SearchExperimentRunsRequest

Optional criteria for searching an experiment's runs. Omit all fields to return the first page of runs in stable `id` ascending order. Keep the filter unchanged when using a returned cursor to fetch the next page.  Filters use the same SQL-like language as span search. Supported column families include unprefixed `id`, `output`, and `example_id`; custom run columns; `eval.<name>.score`, `eval.<name>.label`, `eval.<name>.explanation`, and `eval.<name>.metadata.*`; and `annotation.<name>.*` when present in the run schema. A present empty or whitespace-only filter is invalid. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**filter** | **str** | SQL-like filter expression. Omit to search all runs; an empty or whitespace-only value is invalid. | [optional] 
**limit** | **int** | Maximum number of runs to return. Defaults to 50 and must be between 1 and 500. | [optional] 
**cursor** | **str** | Opaque cursor from &#x60;pagination.next_cursor&#x60;. Omit to start at the first page; keep the filter unchanged while paging. | [optional] 

## Example

```python
from arize._generated.api_client.models.search_experiment_runs_request import SearchExperimentRunsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SearchExperimentRunsRequest from a JSON string
search_experiment_runs_request_instance = SearchExperimentRunsRequest.from_json(json)
# print the JSON string representation of the object
print(SearchExperimentRunsRequest.to_json())

# convert the object into a dict
search_experiment_runs_request_dict = search_experiment_runs_request_instance.to_dict()
# create an instance of SearchExperimentRunsRequest from a dict
search_experiment_runs_request_from_dict = SearchExperimentRunsRequest.from_dict(search_experiment_runs_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


