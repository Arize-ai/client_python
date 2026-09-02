# RemoveTagsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tag_ids** | **List[str]** | IDs of the tags to detach. Up to 100 per request. An ID that is not currently attached is reported in &#x60;not_found&#x60; rather than causing the whole request to fail, so the same request can be retried safely.  | 

## Example

```python
from arize._generated.api_client.models.remove_tags_request import RemoveTagsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of RemoveTagsRequest from a JSON string
remove_tags_request_instance = RemoveTagsRequest.from_json(json)
# print the JSON string representation of the object
print(RemoveTagsRequest.to_json())

# convert the object into a dict
remove_tags_request_dict = remove_tags_request_instance.to_dict()
# create an instance of RemoveTagsRequest from a dict
remove_tags_request_from_dict = RemoveTagsRequest.from_dict(remove_tags_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


