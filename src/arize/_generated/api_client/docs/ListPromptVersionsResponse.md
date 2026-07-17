# ListPromptVersionsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompt_versions** | [**List[PromptVersion]**](PromptVersion.md) | A list of prompt versions | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_prompt_versions_response import ListPromptVersionsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListPromptVersionsResponse from a JSON string
list_prompt_versions_response_instance = ListPromptVersionsResponse.from_json(json)
# print the JSON string representation of the object
print(ListPromptVersionsResponse.to_json())

# convert the object into a dict
list_prompt_versions_response_dict = list_prompt_versions_response_instance.to_dict()
# create an instance of ListPromptVersionsResponse from a dict
list_prompt_versions_response_from_dict = ListPromptVersionsResponse.from_dict(list_prompt_versions_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


