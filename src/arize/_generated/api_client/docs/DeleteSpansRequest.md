# DeleteSpansRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project ID containing the spans to delete | 
**span_ids** | **List[str]** | List of span IDs to delete (maximum 5000) | 

## Example

```python
from arize._generated.api_client.models.delete_spans_request import DeleteSpansRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DeleteSpansRequest from a JSON string
delete_spans_request_instance = DeleteSpansRequest.from_json(json)
# print the JSON string representation of the object
print(DeleteSpansRequest.to_json())

# convert the object into a dict
delete_spans_request_dict = delete_spans_request_instance.to_dict()
# create an instance of DeleteSpansRequest from a dict
delete_spans_request_from_dict = DeleteSpansRequest.from_dict(delete_spans_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


