# WebhookEventType

An event that a webhook can subscribe to. - `PROMPT_VERSION_CREATED` — a new version of the prompt was created. - `PROMPT_VERSION_LABELED` — a label was added to a version of the prompt. - `PROMPT_VERSION_UNLABELED` — a label was removed from a version of the prompt. - `EVALUATOR_VERSION_CREATED` — a new version of the evaluator was created. - `ONLINE_TASK_RUN_FAILED` — a run of an evaluation task failed, or   wrote some evaluations and then stopped on an error. Sent once when   a task starts failing and again each time the cause changes.  Prompt events are valid only for `PROMPT` sources, evaluator events only for `EVALUATOR` sources, and task run events only for `ONLINE_TASK` and `SPACE` sources. 

## Enum

* `PROMPT_VERSION_CREATED` (value: `'PROMPT_VERSION_CREATED'`)

* `PROMPT_VERSION_LABELED` (value: `'PROMPT_VERSION_LABELED'`)

* `PROMPT_VERSION_UNLABELED` (value: `'PROMPT_VERSION_UNLABELED'`)

* `EVALUATOR_VERSION_CREATED` (value: `'EVALUATOR_VERSION_CREATED'`)

* `ONLINE_TASK_RUN_FAILED` (value: `'ONLINE_TASK_RUN_FAILED'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


