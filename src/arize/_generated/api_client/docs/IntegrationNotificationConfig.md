# IntegrationNotificationConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** |  | 
**integration_id** | **str** | The integration to notify (base64 global ID). | 

## Example

```python
from arize._generated.api_client.models.integration_notification_config import IntegrationNotificationConfig

# TODO update the JSON string below
json = "{}"
# create an instance of IntegrationNotificationConfig from a JSON string
integration_notification_config_instance = IntegrationNotificationConfig.from_json(json)
# print the JSON string representation of the object
print(IntegrationNotificationConfig.to_json())

# convert the object into a dict
integration_notification_config_dict = integration_notification_config_instance.to_dict()
# create an instance of IntegrationNotificationConfig from a dict
integration_notification_config_from_dict = IntegrationNotificationConfig.from_dict(integration_notification_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


