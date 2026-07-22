# ScheduledRuntimeConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**enabled** | **bool** | Whether the monitor runs on a schedule. &#x60;false&#x60; means automatic scheduled evaluation is disabled. | 
**cadence_seconds** | **int** | How often the monitor evaluates, in seconds. | [optional] 
**days_of_week** | **List[int]** | Days of the week the monitor runs on (&#x60;0&#x60; &#x3D; Sunday … &#x60;6&#x60; &#x3D; Saturday).  | [optional] 

## Example

```python
from arize._generated.api_client.models.scheduled_runtime_config import ScheduledRuntimeConfig

# TODO update the JSON string below
json = "{}"
# create an instance of ScheduledRuntimeConfig from a JSON string
scheduled_runtime_config_instance = ScheduledRuntimeConfig.from_json(json)
# print the JSON string representation of the object
print(ScheduledRuntimeConfig.to_json())

# convert the object into a dict
scheduled_runtime_config_dict = scheduled_runtime_config_instance.to_dict()
# create an instance of ScheduledRuntimeConfig from a dict
scheduled_runtime_config_from_dict = ScheduledRuntimeConfig.from_dict(scheduled_runtime_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


