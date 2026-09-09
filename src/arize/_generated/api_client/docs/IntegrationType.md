# IntegrationType

The integration category. Selects the shape of `config`. Additive — new types (alerting, webhook, ...) are added non-breakingly.  - `LLM`       — a model-provider integration (e.g. OpenAI). - `AGENT`     — connects your own agent, exposed at an HTTP endpoint. - `EVALUATOR` — connects a remote evaluator endpoint. Only returned when                 `?type=EVALUATOR` is passed explicitly; excluded from the                 default (`LLM` + `AGENT`) list to keep the cursor contract                 stable. Requires the remote evaluators feature to be enabled. 

## Enum

* `LLM` (value: `'LLM'`)

* `AGENT` (value: `'AGENT'`)

* `EVALUATOR` (value: `'EVALUATOR'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


