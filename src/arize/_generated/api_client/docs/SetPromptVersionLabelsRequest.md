# SetPromptVersionLabelsRequest

Labels to set on a prompt version.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**labels** | **List[str]** | Array of label names to set on the version. Replaces all existing labels on this version. Pass an empty array to remove all labels from this version. Labels are unique per prompt — a label can only be assigned to one version at a time. If a label in this array is currently assigned to a different version of the same prompt, it will be moved to this version automatically (no error is raised).  | 

## Example

```python
from arize._generated.api_client.models.set_prompt_version_labels_request import SetPromptVersionLabelsRequest

# TODO update the JSON string below
json = "{}"
# create an instance of SetPromptVersionLabelsRequest from a JSON string
set_prompt_version_labels_request_instance = SetPromptVersionLabelsRequest.from_json(json)
# print the JSON string representation of the object
print(SetPromptVersionLabelsRequest.to_json())

# convert the object into a dict
set_prompt_version_labels_request_dict = set_prompt_version_labels_request_instance.to_dict()
# create an instance of SetPromptVersionLabelsRequest from a dict
set_prompt_version_labels_request_from_dict = SetPromptVersionLabelsRequest.from_dict(set_prompt_version_labels_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


