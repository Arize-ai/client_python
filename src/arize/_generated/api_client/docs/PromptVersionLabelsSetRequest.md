# PromptVersionLabelsSetRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**labels** | **List[str]** | Array of label names to set on the version (replaces all existing labels) | 

## Example

```python
from arize._generated.api_client.models.prompt_version_labels_set_request import PromptVersionLabelsSetRequest

# TODO update the JSON string below
json = "{}"
# create an instance of PromptVersionLabelsSetRequest from a JSON string
prompt_version_labels_set_request_instance = PromptVersionLabelsSetRequest.from_json(json)
# print the JSON string representation of the object
print(PromptVersionLabelsSetRequest.to_json())

# convert the object into a dict
prompt_version_labels_set_request_dict = prompt_version_labels_set_request_instance.to_dict()
# create an instance of PromptVersionLabelsSetRequest from a dict
prompt_version_labels_set_request_from_dict = PromptVersionLabelsSetRequest.from_dict(prompt_version_labels_set_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


