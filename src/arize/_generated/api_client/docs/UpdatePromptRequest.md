# UpdatePromptRequest

Prompt update parameters. At least one field must be provided.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**description** | **str** | Updated description for the prompt | [optional] 

## Example

```python
from arize._generated.api_client.models.update_prompt_request import UpdatePromptRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdatePromptRequest from a JSON string
update_prompt_request_instance = UpdatePromptRequest.from_json(json)
# print the JSON string representation of the object
print(UpdatePromptRequest.to_json())

# convert the object into a dict
update_prompt_request_dict = update_prompt_request_instance.to_dict()
# create an instance of UpdatePromptRequest from a dict
update_prompt_request_from_dict = UpdatePromptRequest.from_dict(update_prompt_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


