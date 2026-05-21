# PromptListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompts** | [**List[Prompt]**](Prompt.md) | A list of prompts | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.prompt_list_response import PromptListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of PromptListResponse from a JSON string
prompt_list_response_instance = PromptListResponse.from_json(json)
# print the JSON string representation of the object
print(PromptListResponse.to_json())

# convert the object into a dict
prompt_list_response_dict = prompt_list_response_instance.to_dict()
# create an instance of PromptListResponse from a dict
prompt_list_response_from_dict = PromptListResponse.from_dict(prompt_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


