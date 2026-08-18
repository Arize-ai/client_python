# ListWebhookDeliveryAttemptsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**delivery_attempts** | [**List[WebhookDeliveryAttempt]**](WebhookDeliveryAttempt.md) | A list of delivery attempts, most recent first | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_webhook_delivery_attempts_response import ListWebhookDeliveryAttemptsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListWebhookDeliveryAttemptsResponse from a JSON string
list_webhook_delivery_attempts_response_instance = ListWebhookDeliveryAttemptsResponse.from_json(json)
# print the JSON string representation of the object
print(ListWebhookDeliveryAttemptsResponse.to_json())

# convert the object into a dict
list_webhook_delivery_attempts_response_dict = list_webhook_delivery_attempts_response_instance.to_dict()
# create an instance of ListWebhookDeliveryAttemptsResponse from a dict
list_webhook_delivery_attempts_response_from_dict = ListWebhookDeliveryAttemptsResponse.from_dict(list_webhook_delivery_attempts_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


