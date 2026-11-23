# CreateTracingMonitorRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Creates a monitor over a project&#39;s tracing data. | 
**project_id** | **str** | The LLM tracing project to monitor (base64 global ID). | 
**name** | **str** | Human-readable name of the monitor. 1 to 255 characters after surrounding whitespace is trimmed.  | 
**metric** | [**TracingDataQualityMetric**](TracingDataQualityMetric.md) | The metric computed over &#x60;dimension&#x60;. Must suit the attribute&#39;s data type. | 
**dimension** | [**TracingAttributeInput**](TracingAttributeInput.md) | The span attribute or LLM eval the metric is computed over. | 
**threshold** | [**ThresholdConfigInput**](ThresholdConfigInput.md) |  | 
**evaluation_window_length_seconds** | **int** | Length of the evaluated window, in seconds: 300, 600, 900, 1800, or a whole number of hours up to 8760 hours.  | [optional] [default to 86400]
**delay_seconds** | **int** | Delay before a window is evaluated, in seconds, so late-arriving spans are counted. 0 to 15552000.  | [optional] [default to 0]
**filters** | [**List[TracingFilterInput]**](TracingFilterInput.md) | Filters narrowing the spans the metric is computed over. At most 25. Defaults to no filters.  | [optional] 
**notification_configs** | [**List[NotificationConfigInput]**](NotificationConfigInput.md) | Destinations notified on a triggered transition. At most 50, with no duplicate destination. Defaults to none.  | [optional] 
**notifications_enabled** | **bool** | Whether notifications fire on a triggered transition. | [optional] [default to True]
**manual_evaluation_enabled** | **bool** | Evaluate the monitor only on demand. Cannot be combined with an enabled &#x60;scheduled_runtime&#x60; or with &#x60;downtime&#x60;.  | [optional] [default to False]
**downtime** | [**DowntimeConfigInput**](DowntimeConfigInput.md) | A recurring pause during which the monitor does not trigger. Defaults to none. | [optional] 
**scheduled_runtime** | [**ScheduledRuntimeConfigInput**](ScheduledRuntimeConfigInput.md) | When the monitor evaluates. Defaults to a disabled schedule, so the monitor evaluates continuously.  | [optional] 
**notes** | **str** | Notes included in notifications, at most 1000 characters. Supports the monitor notes variables; &#x60;{{threshold2.*}}&#x60; variables require a range threshold.  | [optional] 

## Example

```python
from arize._generated.api_client.models.create_tracing_monitor_request import CreateTracingMonitorRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateTracingMonitorRequest from a JSON string
create_tracing_monitor_request_instance = CreateTracingMonitorRequest.from_json(json)
# print the JSON string representation of the object
print(CreateTracingMonitorRequest.to_json())

# convert the object into a dict
create_tracing_monitor_request_dict = create_tracing_monitor_request_instance.to_dict()
# create an instance of CreateTracingMonitorRequest from a dict
create_tracing_monitor_request_from_dict = CreateTracingMonitorRequest.from_dict(create_tracing_monitor_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


