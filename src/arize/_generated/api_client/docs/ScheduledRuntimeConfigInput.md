# ScheduledRuntimeConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**enabled** | **bool** | Whether the monitor evaluates only on the schedule below. | 
**cadence_seconds** | **int** | Interval between evaluations, in seconds: 300, 600, 900, 1800, 1 to 24 hours, 1 to 30 days, 1 to 4 weeks, or 1 to 12 months of 30 days. Required when &#x60;enabled&#x60; is true; defaults to 10800 otherwise.  | [optional] 
**days_of_week** | **List[int]** | Days the monitor may evaluate on, from &#x60;0&#x60; (Sunday) to &#x60;6&#x60; (Saturday). Required when &#x60;enabled&#x60; is true. Defaults to every day when omitted.  | [optional] 

## Example

```python
from arize._generated.api_client.models.scheduled_runtime_config_input import ScheduledRuntimeConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of ScheduledRuntimeConfigInput from a JSON string
scheduled_runtime_config_input_instance = ScheduledRuntimeConfigInput.from_json(json)
# print the JSON string representation of the object
print(ScheduledRuntimeConfigInput.to_json())

# convert the object into a dict
scheduled_runtime_config_input_dict = scheduled_runtime_config_input_instance.to_dict()
# create an instance of ScheduledRuntimeConfigInput from a dict
scheduled_runtime_config_input_from_dict = ScheduledRuntimeConfigInput.from_dict(scheduled_runtime_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


