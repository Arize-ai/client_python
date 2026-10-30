# EvaluatorType

The evaluator type:  - `TEMPLATE` — LLM-based evaluator. - `CODE` — managed built-in evaluators or custom Python code (both are   subtypes of `CODE`, discriminated by the nested `CodeConfig.type` =   `MANAGED` | `CUSTOM`). - `HARNESS` — test harness evaluator. - `REMOTE` — remote evaluator. - `DECISION_JUDGE` — Decision Judge evaluator. It asks a set of typed   questions about each span and writes one result per question. Requires   Decision Judge to be enabled for the account. `GET /v2/evaluators`   returns these evaluators only when `type=DECISION_JUDGE` is passed.  Applies to both the parent `Evaluator.type` field and every version's `type` discriminator — a version's `type` must always match its parent evaluator's `type`. 

## Enum

* `TEMPLATE` (value: `'TEMPLATE'`)

* `CODE` (value: `'CODE'`)

* `HARNESS` (value: `'HARNESS'`)

* `REMOTE` (value: `'REMOTE'`)

* `DECISION_JUDGE` (value: `'DECISION_JUDGE'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


