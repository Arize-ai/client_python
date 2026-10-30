# CreateTypeSafeAiConfig

Create config for a TypeSafe AI integration, used only by Decision Judge evaluators. `api_key` is required and is write-only (never returned; it surfaces as `has_api_key` on read). There is no endpoint field, no custom request headers, and no model list. Creating one returns `400` when this deployment has not enabled the TypeSafe AI provider. This is a deployment setting, separate from the account's Decision Judge access that the evaluator and task endpoints report as `403`.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the TypeSafe AI provider. | 
**api_key** | **str** | TypeSafe AI API key (write-only, never returned). | 

## Example

```python
from arize._generated.api_client.models.create_type_safe_ai_config import CreateTypeSafeAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTypeSafeAiConfig from a JSON string
create_type_safe_ai_config_instance = CreateTypeSafeAiConfig.from_json(json)
# print the JSON string representation of the object
print(CreateTypeSafeAiConfig.to_json())

# convert the object into a dict
create_type_safe_ai_config_dict = create_type_safe_ai_config_instance.to_dict()
# create an instance of CreateTypeSafeAiConfig from a dict
create_type_safe_ai_config_from_dict = CreateTypeSafeAiConfig.from_dict(create_type_safe_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


