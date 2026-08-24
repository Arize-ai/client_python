# AddTagsRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**tag_ids** | **List[str]** | IDs of the tags to attach. Up to 100 per request. Tags must belong to the same space as the resource. Attaching a tag that is already attached is idempotent rather than an error, so the same request can be retried safely.  | 

## Example

```python
from arize._generated.api_client.models.add_tags_request import AddTagsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of AddTagsRequest from a JSON string
add_tags_request_instance = AddTagsRequest.from_json(json)
# print the JSON string representation of the object
print(AddTagsRequest.to_json())

# convert the object into a dict
add_tags_request_dict = add_tags_request_instance.to_dict()
# create an instance of AddTagsRequest from a dict
add_tags_request_from_dict = AddTagsRequest.from_dict(add_tags_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


