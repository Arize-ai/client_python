# WebhookDeliveryAttempt

A single attempt to deliver an event to a webhook's endpoint.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**event_id** | **str** | Unique identifier of the event that triggered this delivery | 
**attempt_number** | **int** | Which attempt this was for the event, starting at 1. Failed deliveries are retried. | 
**payload** | **Dict[str, object]** | The JSON payload sent to the webhook&#39;s endpoint | 
**status_code** | **int** | HTTP status code returned by the endpoint. &#x60;null&#x60; when no response was received. | [optional] 
**error_message** | **str** | Why the delivery failed. &#x60;null&#x60; for successful deliveries. | [optional] 
**created_at** | **datetime** | Timestamp for when the delivery was attempted | 

## Example

```python
from arize._generated.api_client.models.webhook_delivery_attempt import WebhookDeliveryAttempt

# TODO update the JSON string below
json = "{}"
# create an instance of WebhookDeliveryAttempt from a JSON string
webhook_delivery_attempt_instance = WebhookDeliveryAttempt.from_json(json)
# print the JSON string representation of the object
print(WebhookDeliveryAttempt.to_json())

# convert the object into a dict
webhook_delivery_attempt_dict = webhook_delivery_attempt_instance.to_dict()
# create an instance of WebhookDeliveryAttempt from a dict
webhook_delivery_attempt_from_dict = WebhookDeliveryAttempt.from_dict(webhook_delivery_attempt_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


