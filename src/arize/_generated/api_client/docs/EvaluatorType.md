# EvaluatorType

The evaluator type:  - `TEMPLATE` — LLM-based evaluator. - `CODE` — managed built-in evaluators or custom Python code (both are   subtypes of `CODE`, discriminated by the nested `CodeConfig.type` =   `MANAGED` | `CUSTOM`). - `HARNESS` — test harness evaluator. - `REMOTE` — remote evaluator.  Applies to both the parent `Evaluator.type` field and every version's `type` discriminator — a version's `type` must always match its parent evaluator's `type`. 

## Enum

* `TEMPLATE` (value: `'TEMPLATE'`)

* `CODE` (value: `'CODE'`)

* `HARNESS` (value: `'HARNESS'`)

* `REMOTE` (value: `'REMOTE'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


