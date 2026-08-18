# CreateTagRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the tag. Must be unique within the space, compared case-insensitively — a space containing &#x60;Production&#x60; cannot also contain &#x60;production&#x60;. Maximum 100 characters.  | 
**description** | **str** | Description of what the tag is for. Defaults to &#x60;null&#x60; when omitted.  | [optional] 
**color** | [**TagColor**](TagColor.md) | Display color for the tag. Defaults to &#x60;null&#x60; when omitted, meaning no color is assigned.  | [optional] 
**space_id** | **str** | The unique identifier of the space to create the tag in | 

## Example

```python
from arize._generated.api_client.models.create_tag_request import CreateTagRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTagRequest from a JSON string
create_tag_request_instance = CreateTagRequest.from_json(json)
# print the JSON string representation of the object
print(CreateTagRequest.to_json())

# convert the object into a dict
create_tag_request_dict = create_tag_request_instance.to_dict()
# create an instance of CreateTagRequest from a dict
create_tag_request_from_dict = CreateTagRequest.from_dict(create_tag_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


