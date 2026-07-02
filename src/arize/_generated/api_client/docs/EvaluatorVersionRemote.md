# EvaluatorVersionRemote

Evaluator version backed by a remote evaluation config. Only common version metadata (id, commit info, timestamps) is returned — the remote configuration is not yet accessible and will be a future addition. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for this version | 
**evaluator_id** | **str** | The parent evaluator ID | 
**commit_hash** | **str** | A unique hash identifying this version | 
**commit_message** | **str** | A message describing the changes in this version | 
**created_at** | **datetime** | When this version was created | 
**created_by_user_id** | **str** | The unique identifier for the user who created this version | 
**type** | **str** | Discriminator identifying this as a remote evaluator version. | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_remote import EvaluatorVersionRemote

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionRemote from a JSON string
evaluator_version_remote_instance = EvaluatorVersionRemote.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionRemote.to_json())

# convert the object into a dict
evaluator_version_remote_dict = evaluator_version_remote_instance.to_dict()
# create an instance of EvaluatorVersionRemote from a dict
evaluator_version_remote_from_dict = EvaluatorVersionRemote.from_dict(evaluator_version_remote_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


