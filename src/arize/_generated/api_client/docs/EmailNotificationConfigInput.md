# EmailNotificationConfigInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**type** | **str** | Identifies this notification channel as an email address. | 
**email_address** | **str** | A valid email address, at most 254 characters, notified on a triggered transition. | 

## Example

```python
from arize._generated.api_client.models.email_notification_config_input import EmailNotificationConfigInput

# TODO update the JSON string below
json = "{}"
# create an instance of EmailNotificationConfigInput from a JSON string
email_notification_config_input_instance = EmailNotificationConfigInput.from_json(json)
# print the JSON string representation of the object
print(EmailNotificationConfigInput.to_json())

# convert the object into a dict
email_notification_config_input_dict = email_notification_config_input_instance.to_dict()
# create an instance of EmailNotificationConfigInput from a dict
email_notification_config_input_from_dict = EmailNotificationConfigInput.from_dict(email_notification_config_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


