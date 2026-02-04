# PromptsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompts** | [**List[Prompt]**](Prompt.md) | A list of prompts | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.prompts_list200_response import PromptsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of PromptsList200Response from a JSON string
prompts_list200_response_instance = PromptsList200Response.from_json(json)
# print the JSON string representation of the object
print(PromptsList200Response.to_json())

# convert the object into a dict
prompts_list200_response_dict = prompts_list200_response_instance.to_dict()
# create an instance of PromptsList200Response from a dict
prompts_list200_response_from_dict = PromptsList200Response.from_dict(prompts_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


