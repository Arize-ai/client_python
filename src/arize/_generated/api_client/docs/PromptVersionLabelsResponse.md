# PromptVersionLabelsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**labels** | **List[str]** | Label names on the version | 

## Example

```python
from arize._generated.api_client.models.prompt_version_labels_response import PromptVersionLabelsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of PromptVersionLabelsResponse from a JSON string
prompt_version_labels_response_instance = PromptVersionLabelsResponse.from_json(json)
# print the JSON string representation of the object
print(PromptVersionLabelsResponse.to_json())

# convert the object into a dict
prompt_version_labels_response_dict = prompt_version_labels_response_instance.to_dict()
# create an instance of PromptVersionLabelsResponse from a dict
prompt_version_labels_response_from_dict = PromptVersionLabelsResponse.from_dict(prompt_version_labels_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


