# PromptsUpdateRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**description** | **str** | Updated description for the prompt | [optional] 
**tags** | **List[str]** | Updated tags for the prompt | [optional] 

## Example

```python
from arize._generated.api_client.models.prompts_update_request import PromptsUpdateRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PromptsUpdateRequest from a JSON string
prompts_update_request_instance = PromptsUpdateRequest.from_json(json)
# print the JSON string representation of the object
print(PromptsUpdateRequest.to_json())

# convert the object into a dict
prompts_update_request_dict = prompts_update_request_instance.to_dict()
# create an instance of PromptsUpdateRequest from a dict
prompts_update_request_from_dict = PromptsUpdateRequest.from_dict(prompts_update_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


