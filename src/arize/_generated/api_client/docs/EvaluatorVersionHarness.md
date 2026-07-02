# EvaluatorVersionHarness

Evaluator version backed by a harness evaluation config. Only common version metadata (id, commit info, timestamps) is returned — the harness configuration is not yet accessible and will be a future addition. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for this version | 
**evaluator_id** | **str** | The parent evaluator ID | 
**commit_hash** | **str** | A unique hash identifying this version | 
**commit_message** | **str** | A message describing the changes in this version | 
**created_at** | **datetime** | When this version was created | 
**created_by_user_id** | **str** | The unique identifier for the user who created this version | 
**type** | **str** | Discriminator identifying this as a harness evaluator version. | 

## Example

```python
from arize._generated.api_client.models.evaluator_version_harness import EvaluatorVersionHarness

# TODO update the JSON string below
json = "{}"
# create an instance of EvaluatorVersionHarness from a JSON string
evaluator_version_harness_instance = EvaluatorVersionHarness.from_json(json)
# print the JSON string representation of the object
print(EvaluatorVersionHarness.to_json())

# convert the object into a dict
evaluator_version_harness_dict = evaluator_version_harness_instance.to_dict()
# create an instance of EvaluatorVersionHarness from a dict
evaluator_version_harness_from_dict = EvaluatorVersionHarness.from_dict(evaluator_version_harness_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


