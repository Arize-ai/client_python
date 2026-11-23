# TracingAttributeInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**category** | [**TracingAttributeCategory**](TracingAttributeCategory.md) |  | 
**name** | **str** | Exact, case-sensitive name of the attribute as it appears in the project&#39;s recent tracing data. LLM eval names include their scope prefix, for example &#x60;eval.correctness.score&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.tracing_attribute_input import TracingAttributeInput

# TODO update the JSON string below
json = "{}"
# create an instance of TracingAttributeInput from a JSON string
tracing_attribute_input_instance = TracingAttributeInput.from_json(json)
# print the JSON string representation of the object
print(TracingAttributeInput.to_json())

# convert the object into a dict
tracing_attribute_input_dict = tracing_attribute_input_instance.to_dict()
# create an instance of TracingAttributeInput from a dict
tracing_attribute_input_from_dict = TracingAttributeInput.from_dict(tracing_attribute_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


