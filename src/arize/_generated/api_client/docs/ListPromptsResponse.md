# ListPromptsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompts** | [**List[Prompt]**](Prompt.md) | A list of prompts | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_prompts_response import ListPromptsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListPromptsResponse from a JSON string
list_prompts_response_instance = ListPromptsResponse.from_json(json)
# print the JSON string representation of the object
print(ListPromptsResponse.to_json())

# convert the object into a dict
list_prompts_response_dict = list_prompts_response_instance.to_dict()
# create an instance of ListPromptsResponse from a dict
list_prompts_response_from_dict = ListPromptsResponse.from_dict(list_prompts_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


