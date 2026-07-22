# DriftMonitor


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the monitor (base64 global ID). | 
**name** | **str** | Human-readable name of the monitor. | 
**type** | **str** |  | 
**project_id** | **str** | The project that the monitor belongs to (base64 global ID).  | 
**uri** | **str** | The UI deep link to the monitor. | [readonly] 
**status** | [**MonitorStatus**](MonitorStatus.md) | Current evaluation state. Read-only. | 
**threshold** | [**ThresholdConfig**](ThresholdConfig.md) |  | 
**notifications_enabled** | **bool** | Whether notifications fire on a triggered transition. | [default to True]
**manual_evaluation_enabled** | **bool** | Whether the monitor is evaluated manually rather than on the automatic cadence. | [default to False]
**created_by_user_id** | **str** | The user who created the monitor (base64 global ID). | 
**filters** | [**List[MonitorFilter]**](MonitorFilter.md) | Data filters applied to the monitor&#39;s metric. Omitted when the metric is computed over all data.  | [optional] 
**notification_configs** | [**List[NotificationConfig]**](NotificationConfig.md) | Notification channels (email / integration / webhook) notified on a triggered transition. Omitted when no channels are configured.  | [optional] 
**created_at** | **datetime** | When the monitor was created. | [readonly] 
**updated_at** | **datetime** | When the monitor was last updated. | [readonly] 
**downtime** | [**DowntimeConfig**](DowntimeConfig.md) |  | [optional] 
**scheduled_runtime** | [**ScheduledRuntimeConfig**](ScheduledRuntimeConfig.md) |  | [optional] 
**evaluation_window_length_seconds** | **int** | The length of the evaluation window in seconds. Omitted when unset.  | [optional] 
**delay_seconds** | **int** | The delay applied before evaluating a window, in seconds. Omitted when unset.  | [optional] 
**evaluated_at** | **datetime** | The last time the metric was computed. Omitted if it has never been evaluated. | [optional] [readonly] 
**latest_computed_value** | **float** | The most recently computed metric value. Omitted if it has never been evaluated. | [optional] [readonly] 
**notes** | **str** | Free-form notes attached to the monitor. Omitted when unset. | [optional] 
**metric** | [**DriftMetric**](DriftMetric.md) |  | 
**dimension** | [**Dimension**](Dimension.md) |  | 
**baseline_config** | [**BaselineConfig**](BaselineConfig.md) |  | 
**model_versions** | **List[str]** | The model versions the metric is scoped to. An empty array means all model versions.  | 

## Example

```python
from arize._generated.api_client.models.drift_monitor import DriftMonitor

# TODO update the JSON string below
json = "{}"
# create an instance of DriftMonitor from a JSON string
drift_monitor_instance = DriftMonitor.from_json(json)
# print the JSON string representation of the object
print(DriftMonitor.to_json())

# convert the object into a dict
drift_monitor_dict = drift_monitor_instance.to_dict()
# create an instance of DriftMonitor from a dict
drift_monitor_from_dict = DriftMonitor.from_dict(drift_monitor_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


