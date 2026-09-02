# RemoveTagsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**removed** | **List[str]** | IDs of the tags that were attached and have been detached. | 
**not_found** | **List[str]** | IDs from the request that were not attached to the resource. Not an error — detaching an already-detached tag is a no-op.  | 

## Example

```python
from arize._generated.api_client.models.remove_tags_response import RemoveTagsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of RemoveTagsResponse from a JSON string
remove_tags_response_instance = RemoveTagsResponse.from_json(json)
# print the JSON string representation of the object
print(RemoveTagsResponse.to_json())

# convert the object into a dict
remove_tags_response_dict = remove_tags_response_instance.to_dict()
# create an instance of RemoveTagsResponse from a dict
remove_tags_response_from_dict = RemoveTagsResponse.from_dict(remove_tags_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


