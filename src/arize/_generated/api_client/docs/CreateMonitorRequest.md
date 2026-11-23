# CreateMonitorRequest

The monitor to create. `type` selects the variant: `TRACING` on an LLM tracing project, or `DATA_QUALITY` on an ML project. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Creates a monitor over a project&#39;s tracing data. | 
**project_id** | **str** | The ML project to monitor (base64 global ID). | 
**name** | **str** | Human-readable name of the monitor. 1 to 255 characters after surrounding whitespace is trimmed.  | 
**metric** | [**DataQualityMetric**](DataQualityMetric.md) | The metric computed over &#x60;dimension&#x60;. Numeric dimensions support &#x60;COUNT&#x60;, &#x60;PERCENT_EMPTY&#x60;, &#x60;AVG&#x60;, &#x60;SUM&#x60;, &#x60;STANDARD_DEVIATION&#x60;, &#x60;P50&#x60;, &#x60;P95&#x60;, &#x60;P99&#x60; and &#x60;P99_9&#x60;. Categorical dimensions support &#x60;COUNT&#x60;, &#x60;PERCENT_EMPTY&#x60;, &#x60;CARDINALITY&#x60;, &#x60;NEW_VALUES&#x60; and &#x60;MISSING_VALUES&#x60;. String-list dimensions support &#x60;AVERAGE_STRING_LIST_LENGTH&#x60;, &#x60;COUNT&#x60; and &#x60;PERCENT_EMPTY&#x60;. &#x60;NEW_VALUES&#x60; and &#x60;MISSING_VALUES&#x60; require &#x60;baseline_config&#x60;.  | 
**dimension** | [**MlDimensionInput**](MlDimensionInput.md) | The field the metric is computed over. Embedding features cannot be monitored. | 
**threshold** | [**ThresholdConfigInput**](ThresholdConfigInput.md) |  | 
**evaluation_window_length_seconds** | **int** | Length of the evaluated window, in seconds: 300, 600, 900, 1800, or a whole number of hours up to 8760 hours.  | [optional] [default to 86400]
**delay_seconds** | **int** | Delay before a window is evaluated, in seconds, so late-arriving data is counted. 0 to 15552000.  | [optional] [default to 0]
**filters** | [**List[MlFilterInput]**](MlFilterInput.md) | Filters narrowing the data the metric is computed over. At most 25. Defaults to no filters.  | [optional] 
**notification_configs** | [**List[NotificationConfigInput]**](NotificationConfigInput.md) | Destinations notified on a triggered transition. At most 50, with no duplicate destination. Defaults to none.  | [optional] 
**notifications_enabled** | **bool** | Whether notifications fire on a triggered transition. | [optional] [default to True]
**manual_evaluation_enabled** | **bool** | Evaluate the monitor only on demand. Cannot be combined with an enabled &#x60;scheduled_runtime&#x60; or with &#x60;downtime&#x60;.  | [optional] [default to False]
**downtime** | [**DowntimeConfigInput**](DowntimeConfigInput.md) | A recurring pause during which the monitor does not trigger. Defaults to none. | [optional] 
**scheduled_runtime** | [**ScheduledRuntimeConfigInput**](ScheduledRuntimeConfigInput.md) | When the monitor evaluates. Defaults to a disabled schedule, so the monitor evaluates continuously.  | [optional] 
**notes** | **str** | Notes included in notifications, at most 1000 characters. Supports the monitor notes variables; &#x60;{{threshold2.*}}&#x60; variables require a range threshold.  | [optional] 
**environment** | [**MlMonitorEnvironment**](MlMonitorEnvironment.md) | The environment whose data is monitored. | [optional] 
**model_versions** | **List[str]** | External model version IDs the data is limited to, at most 100 with no duplicates, each an existing version of the project. Defaults to all versions.  | [optional] 
**baseline_config** | [**BaselineConfigInput**](BaselineConfigInput.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_monitor_request import CreateMonitorRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateMonitorRequest from a JSON string
create_monitor_request_instance = CreateMonitorRequest.from_json(json)
# print the JSON string representation of the object
print(CreateMonitorRequest.to_json())

# convert the object into a dict
create_monitor_request_dict = create_monitor_request_instance.to_dict()
# create an instance of CreateMonitorRequest from a dict
create_monitor_request_from_dict = CreateMonitorRequest.from_dict(create_monitor_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


