# SearchDatasetExamplesRequest

Optional criteria for searching a dataset's examples. Omit all fields to return the first page of examples in ascending order of `created_at`, with `id` as a tiebreaker. Keep the filter unchanged when using a returned cursor to fetch the next page.  Filters use the same SQL-like language as span search. Supported column families include unprefixed example columns, `id`, and `annotation.<name>.*` when present in the example schema. A present empty or whitespace-only filter is invalid. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**filter** | **str** | SQL-like filter expression. Omit to search all examples; an empty or whitespace-only value is invalid. | [optional] 
**limit** | **int** | Maximum number of examples to return. Defaults to 50 and must be between 1 and 500. | [optional] 
**cursor** | **str** | Opaque cursor from &#x60;pagination.next_cursor&#x60;. Omit to start at the first page; keep the filter unchanged while paging. | [optional] 
**dataset_version_id** | **str** | Unique identifier of the dataset version to search. If omitted, the latest version is selected. | [optional] 

## Example

```python
from arize._generated.api_client.models.search_dataset_examples_request import SearchDatasetExamplesRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SearchDatasetExamplesRequest from a JSON string
search_dataset_examples_request_instance = SearchDatasetExamplesRequest.from_json(json)
# print the JSON string representation of the object
print(SearchDatasetExamplesRequest.to_json())

# convert the object into a dict
search_dataset_examples_request_dict = search_dataset_examples_request_instance.to_dict()
# create an instance of SearchDatasetExamplesRequest from a dict
search_dataset_examples_request_from_dict = SearchDatasetExamplesRequest.from_dict(search_dataset_examples_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


