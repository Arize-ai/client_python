# SpansDelete200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**deleted_span_ids** | **List[str]** | Span IDs confirmed deleted across all successfully processed intervals. | 

## Example

```python
from arize._generated.api_client.models.spans_delete200_response import SpansDelete200Response

# TODO update the JSON string below
json = "{}"
# create an instance of SpansDelete200Response from a JSON string
spans_delete200_response_instance = SpansDelete200Response.from_json(json)
# print the JSON string representation of the object
print(SpansDelete200Response.to_json())

# convert the object into a dict
spans_delete200_response_dict = spans_delete200_response_instance.to_dict()
# create an instance of SpansDelete200Response from a dict
spans_delete200_response_from_dict = SpansDelete200Response.from_dict(spans_delete200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


