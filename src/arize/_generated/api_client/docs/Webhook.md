# Webhook

A webhook is an organization-owned destination that receives event deliveries over HTTPS. Attach a webhook to prompts and evaluators through their webhook-subscription endpoints to choose which events it receives.  Credentials are write-only: the bearer token is never returned, and the HMAC signing secret is returned exactly once, in the create response — only its redacted hint is readable afterwards. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | Unique identifier for the webhook | 
**organization_id** | **str** | The unique identifier of the organization that owns the webhook | 
**name** | **str** | Name of the webhook (unique within the organization) | 
**description** | **str** | A brief description of the webhook&#39;s purpose. Defaults to an empty string. | 
**url** | **str** | The HTTPS endpoint events are delivered to | 
**auth_type** | [**WebhookAuthType**](WebhookAuthType.md) | How deliveries from this webhook are authenticated. Fixed at creation. | 
**signing_secret_hint** | **str** | Redacted hint of the signing secret (e.g. &#x60;whsec_…abcd&#x60;), useful for identifying which secret the webhook uses. Present only for &#x60;HMAC_SHA256&#x60; webhooks. The full secret is returned exactly once, in the create response, and cannot be retrieved afterwards.  | [optional] 
**timeout_ms** | **int** | How long a delivery request may run before it is abandoned, in milliseconds. Defaults to 30000. | 
**headers** | **Dict[str, str]** | Custom HTTP headers sent with each delivery request | 
**created_at** | **datetime** | Timestamp for when the webhook was created | 
**updated_at** | **datetime** | Timestamp for when the webhook was last updated | 
**created_by_user_id** | **str** | The unique identifier of the user who created the webhook. Absent when that user has since been removed from the account. | [optional] 

## Example

```python
from arize._generated.api_client.models.webhook import Webhook

# TODO update the JSON string below
json = "{}"
# create an instance of Webhook from a JSON string
webhook_instance = Webhook.from_json(json)
# print the JSON string representation of the object
print(Webhook.to_json())

# convert the object into a dict
webhook_dict = webhook_instance.to_dict()
# create an instance of Webhook from a dict
webhook_from_dict = Webhook.from_dict(webhook_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


