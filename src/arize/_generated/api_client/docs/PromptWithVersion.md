# PromptWithVersion

A prompt with a resolved version. Returned by Create Prompt and Get Prompt. The version is the initial version on create, or the resolved version (latest, by ID, or by label) on get. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The prompt ID | 
**name** | **str** | The prompt name | 
**description** | **str** | The prompt description | [optional] 
**space_id** | **str** | The space ID the prompt belongs to | 
**created_at** | **datetime** | When the prompt was created | 
**updated_at** | **datetime** | When the prompt was last updated | 
**created_by_user_id** | **str** | The user ID of the user who created the prompt | 
**version** | [**PromptVersion**](PromptVersion.md) |  | 

## Example

```python
from arize._generated.api_client.models.prompt_with_version import PromptWithVersion

# TODO update the JSON string below
json = "{}"
# create an instance of PromptWithVersion from a JSON string
prompt_with_version_instance = PromptWithVersion.from_json(json)
# print the JSON string representation of the object
print(PromptWithVersion.to_json())

# convert the object into a dict
prompt_with_version_dict = prompt_with_version_instance.to_dict()
# create an instance of PromptWithVersion from a dict
prompt_with_version_from_dict = PromptWithVersion.from_dict(prompt_with_version_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


