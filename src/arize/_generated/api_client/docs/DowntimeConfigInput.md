# DowntimeConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**start** | **datetime** | When the first downtime window begins, as an RFC 3339 timestamp with an offset. | 
**duration_seconds** | **int** | Length of each downtime window, in seconds. A whole number of hours from 1 to 1344, and no longer than &#x60;frequency_days&#x60;.  | 
**frequency_days** | **int** | Days between the starts of successive downtime windows, from 1 to 56. | 

## Example

```python
from arize._generated.api_client.models.downtime_config_input import DowntimeConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of DowntimeConfigInput from a JSON string
downtime_config_input_instance = DowntimeConfigInput.from_json(json)
# print the JSON string representation of the object
print(DowntimeConfigInput.to_json())

# convert the object into a dict
downtime_config_input_dict = downtime_config_input_instance.to_dict()
# create an instance of DowntimeConfigInput from a dict
downtime_config_input_from_dict = DowntimeConfigInput.from_dict(downtime_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


