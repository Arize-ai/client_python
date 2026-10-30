# TypeSafeAiConfig

Config for a TypeSafe AI integration. TypeSafe AI integrations are used only by Decision Judge evaluators and cannot be used in prompts, experiments, or other features. TypeSafe AI is a single hosted service, so there is no endpoint field, no custom request headers, and no model list. The API key is write-only and surfaces as `has_api_key`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the TypeSafe AI provider. | 
**has_api_key** | **bool** | Whether an API key is configured (the key itself is never returned). | 

## Example

```python
from arize._generated.api_client.models.type_safe_ai_config import TypeSafeAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of TypeSafeAiConfig from a JSON string
type_safe_ai_config_instance = TypeSafeAiConfig.from_json(json)
# print the JSON string representation of the object
print(TypeSafeAiConfig.to_json())

# convert the object into a dict
type_safe_ai_config_dict = type_safe_ai_config_instance.to_dict()
# create an instance of TypeSafeAiConfig from a dict
type_safe_ai_config_from_dict = TypeSafeAiConfig.from_dict(type_safe_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


