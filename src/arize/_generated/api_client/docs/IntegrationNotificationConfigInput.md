# IntegrationNotificationConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this notification channel as a connected alert integration. | 
**integration_id** | **str** | The alert integration to notify (base64 global ID). It must belong to the project&#39;s organization. | 

## Example

```python
from arize._generated.api_client.models.integration_notification_config_input import IntegrationNotificationConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of IntegrationNotificationConfigInput from a JSON string
integration_notification_config_input_instance = IntegrationNotificationConfigInput.from_json(json)
# print the JSON string representation of the object
print(IntegrationNotificationConfigInput.to_json())

# convert the object into a dict
integration_notification_config_input_dict = integration_notification_config_input_instance.to_dict()
# create an instance of IntegrationNotificationConfigInput from a dict
integration_notification_config_input_from_dict = IntegrationNotificationConfigInput.from_dict(integration_notification_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


