# WebhookAuthType

How deliveries from this webhook are authenticated. - `BEARER`: the stored `auth_token` is sent verbatim as the   `Authorization` header of each delivery request. - `HMAC_SHA256`: each delivery is signed with the webhook's signing   secret. The `X-Arize-Webhook-Signature` header carries   `v1=<hex-encoded HMAC-SHA256>` computed over   `<timestamp>.<raw request body>`, where `<timestamp>` is the   Unix-seconds value from the `X-Arize-Webhook-Timestamp` header and the   raw body is the exact bytes received. Deliveries also carry   `X-Arize-Webhook-Id` (event identifier) and `X-Arize-Webhook-Event`   (event type). To verify, recompute the HMAC over the received   timestamp and raw body with your stored secret and compare it to the   signature. 

## Enum

* `BEARER` (value: `'BEARER'`)

* `HMAC_SHA256` (value: `'HMAC_SHA256'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


