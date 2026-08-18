# CreateWebhookResponse

The created webhook, plus `signing_secret` for `HMAC_SHA256` webhooks — the only time the secret is ever returned. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the webhook | 
**organization_id** | **str** | The unique identifier of the organization that owns the webhook | 
**name** | **str** | Name of the webhook (unique within the organization) | 
**description** | **str** | A brief description of the webhook&#39;s purpose. Defaults to an empty string. | 
**url** | **str** | The HTTPS endpoint events are delivered to | 
**auth_type** | [**WebhookAuthType**](WebhookAuthType.md) | How deliveries from this webhook are authenticated. Fixed at creation. | 
**signing_secret** | **str** | The secret used to verify delivery signatures. **Only returned once**, in this response, when &#x60;auth_type&#x60; is &#x60;HMAC_SHA256&#x60;. Store it securely — it cannot be retrieved again; only a redacted hint (&#x60;signing_secret_hint&#x60;) is readable afterwards. Absent for &#x60;BEARER&#x60; webhooks.  | [optional] 
**signing_secret_hint** | **str** | Redacted hint of the signing secret (e.g. &#x60;whsec_…abcd&#x60;), useful for identifying which secret the webhook uses. Present only for &#x60;HMAC_SHA256&#x60; webhooks.  | [optional] 
**timeout_ms** | **int** | How long a delivery request may run before it is abandoned, in milliseconds. Defaults to 30000. | 
**headers** | **Dict[str, str]** | Custom HTTP headers sent with each delivery request | 
**created_at** | **datetime** | Timestamp for when the webhook was created | 
**updated_at** | **datetime** | Timestamp for when the webhook was last updated | 
**created_by_user_id** | **str** | The unique identifier of the user who created the webhook. Absent when that user has since been removed from the account. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_webhook_response import CreateWebhookResponse

# TODO update the JSON string below
json = "{}"
# create an instance of CreateWebhookResponse from a JSON string
create_webhook_response_instance = CreateWebhookResponse.from_json(json)
# print the JSON string representation of the object
print(CreateWebhookResponse.to_json())

# convert the object into a dict
create_webhook_response_dict = create_webhook_response_instance.to_dict()
# create an instance of CreateWebhookResponse from a dict
create_webhook_response_from_dict = CreateWebhookResponse.from_dict(create_webhook_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


