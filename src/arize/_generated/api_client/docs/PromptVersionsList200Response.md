# PromptVersionsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**prompt_versions** | [**List[PromptVersion]**](PromptVersion.md) | A list of prompt versions | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.prompt_versions_list200_response import PromptVersionsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of PromptVersionsList200Response from a JSON string
prompt_versions_list200_response_instance = PromptVersionsList200Response.from_json(json)
# print the JSON string representation of the object
print(PromptVersionsList200Response.to_json())

# convert the object into a dict
prompt_versions_list200_response_dict = prompt_versions_list200_response_instance.to_dict()
# create an instance of PromptVersionsList200Response from a dict
prompt_versions_list200_response_from_dict = PromptVersionsList200Response.from_dict(prompt_versions_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


