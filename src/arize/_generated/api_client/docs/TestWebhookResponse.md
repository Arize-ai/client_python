# TestWebhookResponse

The outcome of a test delivery to the webhook's endpoint.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**status_code** | **int** | HTTP status code returned by the webhook&#39;s endpoint. &#x60;502&#x60; when no response was received, for example because the endpoint was unreachable or timed out. | 
**error_message** | **str** | Why the test delivery failed. &#x60;null&#x60; for successful deliveries. | 

## Example

```python
from arize._generated.api_client.models.test_webhook_response import TestWebhookResponse

# TODO update the JSON string below
json = "{}"
# create an instance of TestWebhookResponse from a JSON string
test_webhook_response_instance = TestWebhookResponse.from_json(json)
# print the JSON string representation of the object
print(TestWebhookResponse.to_json())

# convert the object into a dict
test_webhook_response_dict = test_webhook_response_instance.to_dict()
# create an instance of TestWebhookResponse from a dict
test_webhook_response_from_dict = TestWebhookResponse.from_dict(test_webhook_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


