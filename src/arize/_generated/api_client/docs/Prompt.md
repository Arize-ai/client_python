# Prompt

A prompt is a reusable template for LLM interactions. Prompts can be versioned and labeled to track changes over time. Use prompts to standardize how you interact with LLMs across your application. 

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
**tags** | **List[str]** | The tags associated with the prompt | [optional] 

## Example

```python
from arize._generated.api_client.models.prompt import Prompt

# TODO update the JSON string below
json = "{}"
# create an instance of Prompt from a JSON string
prompt_instance = Prompt.from_json(json)
# print the JSON string representation of the object
print(Prompt.to_json())

# convert the object into a dict
prompt_dict = prompt_instance.to_dict()
# create an instance of Prompt from a dict
prompt_from_dict = Prompt.from_dict(prompt_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


