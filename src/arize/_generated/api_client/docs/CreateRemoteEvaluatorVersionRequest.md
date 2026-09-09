# CreateRemoteEvaluatorVersionRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**commit_message** | **str** | Commit message describing the changes | 
**remote_config** | [**RemoteConfigInput**](RemoteConfigInput.md) |  | 

## Example

```python
from arize._generated.api_client.models.create_remote_evaluator_version_request import CreateRemoteEvaluatorVersionRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateRemoteEvaluatorVersionRequest from a JSON string
create_remote_evaluator_version_request_instance = CreateRemoteEvaluatorVersionRequest.from_json(json)
# print the JSON string representation of the object
print(CreateRemoteEvaluatorVersionRequest.to_json())

# convert the object into a dict
create_remote_evaluator_version_request_dict = create_remote_evaluator_version_request_instance.to_dict()
# create an instance of CreateRemoteEvaluatorVersionRequest from a dict
create_remote_evaluator_version_request_from_dict = CreateRemoteEvaluatorVersionRequest.from_dict(create_remote_evaluator_version_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


