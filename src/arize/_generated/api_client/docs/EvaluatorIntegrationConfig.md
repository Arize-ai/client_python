# EvaluatorIntegrationConfig

Configuration for `type: EVALUATOR` integrations: a customer-hosted HTTPS endpoint plus a JSON Schema describing the request payload. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**endpoint** | **str** | HTTPS endpoint URL Arize calls for remote evaluation. Validated server-side for SSRF (must resolve to a public address).  | 
**has_headers** | **bool** | Whether any headers are configured. Read-only — derived from &#x60;headers&#x60; on write. Header values are never returned.  | [readonly] 
**input_schema** | **Dict[str, object]** | JSON Schema (Draft-07) the endpoint&#39;s request body conforms to. The root schema must have &#x60;type: object&#x60;, must not define the reserved top-level &#x60;arize_metadata&#x60; field, and must not exceed 64 KiB.  | 

## Example

```python
from arize._generated.api_client.models.evaluator_integration_config import EvaluatorIntegrationConfig

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorIntegrationConfig from a JSON string
evaluator_integration_config_instance = EvaluatorIntegrationConfig.from_json(json)
# print the JSON string representation of the object
print(EvaluatorIntegrationConfig.to_json())

# convert the object into a dict
evaluator_integration_config_dict = evaluator_integration_config_instance.to_dict()
# create an instance of EvaluatorIntegrationConfig from a dict
evaluator_integration_config_from_dict = EvaluatorIntegrationConfig.from_dict(evaluator_integration_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


