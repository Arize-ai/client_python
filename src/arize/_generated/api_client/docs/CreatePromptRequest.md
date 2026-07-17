# CreatePromptRequest

Prompt creation parameters with an initial version.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | ID of the space to create the prompt in | 
**name** | **str** | Name of the prompt (must be unique within the space) | 
**description** | **str** | Description of the prompt. Optional. If omitted, the prompt has no description. | [optional] 
**version** | [**PromptVersionCreateRequest**](PromptVersionCreateRequest.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_prompt_request import CreatePromptRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreatePromptRequest from a JSON string
create_prompt_request_instance = CreatePromptRequest.from_json(json)
# print the JSON string representation of the object
print(CreatePromptRequest.to_json())

# convert the object into a dict
create_prompt_request_dict = create_prompt_request_instance.to_dict()
# create an instance of CreatePromptRequest from a dict
create_prompt_request_from_dict = CreatePromptRequest.from_dict(create_prompt_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


