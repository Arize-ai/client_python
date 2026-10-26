# WebhookSourceType

The kind of resource a webhook subscription is attached to. - `PROMPT` — `source_id` is a prompt id. Prompt events apply. - `EVALUATOR` — `source_id` is an evaluator id. Evaluator events apply. - `ONLINE_TASK` — `source_id` is an evaluation task id. Task run events apply. - `SPACE` — `source_id` is a space id. Task run events apply to every   evaluation task in the space, including tasks created later. 

## Enum

* `PROMPT` (value: `'PROMPT'`)

* `EVALUATOR` (value: `'EVALUATOR'`)

* `ONLINE_TASK` (value: `'ONLINE_TASK'`)

* `SPACE` (value: `'SPACE'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


