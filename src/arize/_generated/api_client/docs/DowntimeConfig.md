# DowntimeConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**start** | **datetime** | When the downtime window begins. | 
**duration_seconds** | **int** | How long each downtime window lasts, in seconds. | 
**frequency_days** | **int** | How often the downtime window repeats, in days. | 

## Example

```python
from arize._generated.api_client.models.downtime_config import DowntimeConfig

# TODO update the JSON string below
json = "{}"
# create an instance of DowntimeConfig from a JSON string
downtime_config_instance = DowntimeConfig.from_json(json)
# print the JSON string representation of the object
print(DowntimeConfig.to_json())

# convert the object into a dict
downtime_config_dict = downtime_config_instance.to_dict()
# create an instance of DowntimeConfig from a dict
downtime_config_from_dict = DowntimeConfig.from_dict(downtime_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


