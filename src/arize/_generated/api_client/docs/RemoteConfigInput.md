# RemoteConfigInput

Remote configuration for a `REMOTE` evaluator version in write requests. The referenced integration may also be used by other versions; editing it affects every version that references it. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**integration_id** | **str** | &#x60;EVALUATOR&#x60; integration identifier (base64), as returned by the integrations API (&#x60;POST /v2/integrations&#x60; with &#x60;type: EVALUATOR&#x60;). Must reference an integration of type &#x60;EVALUATOR&#x60;; other integration types are rejected.  | 

## Example

```python
from arize._generated.api_client.models.remote_config_input import RemoteConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of RemoteConfigInput from a JSON string
remote_config_input_instance = RemoteConfigInput.from_json(json)
# print the JSON string representation of the object
print(RemoteConfigInput.to_json())

# convert the object into a dict
remote_config_input_dict = remote_config_input_instance.to_dict()
# create an instance of RemoteConfigInput from a dict
remote_config_input_from_dict = RemoteConfigInput.from_dict(remote_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


