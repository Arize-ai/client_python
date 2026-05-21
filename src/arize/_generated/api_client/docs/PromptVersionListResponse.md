# PromptVersionListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompt_versions** | [**List[PromptVersion]**](PromptVersion.md) | A list of prompt versions | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.prompt_version_list_response import PromptVersionListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of PromptVersionListResponse from a JSON string
prompt_version_list_response_instance = PromptVersionListResponse.from_json(json)
# print the JSON string representation of the object
print(PromptVersionListResponse.to_json())

# convert the object into a dict
prompt_version_list_response_dict = prompt_version_list_response_instance.to_dict()
# create an instance of PromptVersionListResponse from a dict
prompt_version_list_response_from_dict = PromptVersionListResponse.from_dict(prompt_version_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


