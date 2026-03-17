# PromptsCreateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**space_id** | **str** | ID of the space to create the prompt in | 
**name** | **str** | Name of the prompt (must be unique within the space) | 
**description** | **str** | Description of the prompt. Optional. If omitted, the prompt has no description. | [optional] 
**version** | [**PromptVersionCreateRequest**](PromptVersionCreateRequest.md) |  | 

## Example

```python
from arize._generated.api_client.models.prompts_create_request import PromptsCreateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PromptsCreateRequest from a JSON string
prompts_create_request_instance = PromptsCreateRequest.from_json(json)
# print the JSON string representation of the object
print(PromptsCreateRequest.to_json())

# convert the object into a dict
prompts_create_request_dict = prompts_create_request_instance.to_dict()
# create an instance of PromptsCreateRequest from a dict
prompts_create_request_from_dict = PromptsCreateRequest.from_dict(prompts_create_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


