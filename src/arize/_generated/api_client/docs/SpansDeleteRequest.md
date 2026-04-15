# SpansDeleteRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**project_id** | **str** | The project ID containing the spans to delete | 
**span_ids** | **List[str]** | List of span IDs to delete (maximum 1000) | 

## Example

```python
from arize._generated.api_client.models.spans_delete_request import SpansDeleteRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SpansDeleteRequest from a JSON string
spans_delete_request_instance = SpansDeleteRequest.from_json(json)
# print the JSON string representation of the object
print(SpansDeleteRequest.to_json())

# convert the object into a dict
spans_delete_request_dict = spans_delete_request_instance.to_dict()
# create an instance of SpansDeleteRequest from a dict
spans_delete_request_from_dict = SpansDeleteRequest.from_dict(spans_delete_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


