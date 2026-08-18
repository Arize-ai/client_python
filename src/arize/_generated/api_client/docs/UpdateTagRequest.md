# UpdateTagRequest

Fields to update on a tag. Omitted fields are left unchanged, so at least one field must be provided. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New tag name. Must be unique within the space, compared case-insensitively. Maximum 100 characters. Left unchanged when omitted.  | [optional] 
**description** | **str** | New description. Pass &#x60;null&#x60; to clear it. Left unchanged when omitted.  | [optional] 
**color** | [**TagColor**](TagColor.md) | New display color. Pass &#x60;null&#x60; to clear it. Left unchanged when omitted.  | [optional] 

## Example

```python
from arize._generated.api_client.models.update_tag_request import UpdateTagRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateTagRequest from a JSON string
update_tag_request_instance = UpdateTagRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateTagRequest.to_json())

# convert the object into a dict
update_tag_request_dict = update_tag_request_instance.to_dict()
# create an instance of UpdateTagRequest from a dict
update_tag_request_from_dict = UpdateTagRequest.from_dict(update_tag_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


