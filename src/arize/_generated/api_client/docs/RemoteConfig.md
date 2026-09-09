# RemoteConfig

Remote configuration for a `REMOTE` evaluator version. The backing `EVALUATOR` integration is referenced by this version and may also be referenced by other versions. Updating it affects every version that references it.  `integration_id` is included only when the caller has permission to read the backing integration. Callers without that permission still receive the remote version metadata, but the integration reference is omitted. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**integration_id** | **str** | &#x60;EVALUATOR&#x60; integration identifier (base64), as returned by the integrations API (&#x60;POST /v2/integrations&#x60; with &#x60;type: EVALUATOR&#x60;). Must reference an integration of type &#x60;EVALUATOR&#x60;; other integration types are rejected.  | [optional] 

## Example

```python
from arize._generated.api_client.models.remote_config import RemoteConfig

# TODO update the JSON string below
json = "{}"
# create an instance of RemoteConfig from a JSON string
remote_config_instance = RemoteConfig.from_json(json)
# print the JSON string representation of the object
print(RemoteConfig.to_json())

# convert the object into a dict
remote_config_dict = remote_config_instance.to_dict()
# create an instance of RemoteConfig from a dict
remote_config_from_dict = RemoteConfig.from_dict(remote_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


