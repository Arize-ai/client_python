# PromptVersionLabelsSetRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**labels** | **List[str]** | Array of label names to set on the version. Replaces all existing labels on this version. Pass an empty array to remove all labels from this version. Labels are unique per prompt — a label can only be assigned to one version at a time. If a label in this array is currently assigned to a different version of the same prompt, it will be moved to this version automatically (no error is raised).  | 

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


