# ListEvaluatorTemplatesResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**evaluator_templates** | [**List[EvaluatorTemplate]**](EvaluatorTemplate.md) | Every built-in template, ordered by category as the product presents them (response quality, code quality, trajectory, RAG, security, session).  | 

## Example

```python
from arize._generated.api_client.models.list_evaluator_templates_response import ListEvaluatorTemplatesResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListEvaluatorTemplatesResponse from a JSON string
list_evaluator_templates_response_instance = ListEvaluatorTemplatesResponse.from_json(json)
# print the JSON string representation of the object
print(ListEvaluatorTemplatesResponse.to_json())

# convert the object into a dict
list_evaluator_templates_response_dict = list_evaluator_templates_response_instance.to_dict()
# create an instance of ListEvaluatorTemplatesResponse from a dict
list_evaluator_templates_response_from_dict = ListEvaluatorTemplatesResponse.from_dict(list_evaluator_templates_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


