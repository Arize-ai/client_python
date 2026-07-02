# LlmIntegration

An LLM integration (type=llm).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The integration ID (base64 global ID). | 
**type** | **str** | Discriminator identifying an LLM integration. | 
**name** | **str** | The integration name. Unique per (account, type). | 
**scopings** | [**List[IntegrationScoping]**](IntegrationScoping.md) | Visibility scoping rules. Account-wide when empty. | 
**created_at** | **datetime** | When the integration was created. | 
**updated_at** | **datetime** | When the integration was last updated. | 
**created_by_user_id** | **str** | Global ID of the user who created the integration. | 
**config** | [**OpenAiConfig**](OpenAiConfig.md) |  | 

## Example

```python
from arize._generated.api_client.models.llm_integration import LlmIntegration

# TODO update the JSON string below
json = "{}"
# create an instance of LlmIntegration from a JSON string
llm_integration_instance = LlmIntegration.from_json(json)
# print the JSON string representation of the object
print(LlmIntegration.to_json())

# convert the object into a dict
llm_integration_dict = llm_integration_instance.to_dict()
# create an instance of LlmIntegration from a dict
llm_integration_from_dict = LlmIntegration.from_dict(llm_integration_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


