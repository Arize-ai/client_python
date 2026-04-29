# EvaluatorType

The evaluator type: `template` (LLM-based) or `code` (managed built-in evaluators or custom Python code — both are subtypes of `code`, discriminated by the nested `CodeConfig.type` = `managed` | `custom`). Applies to both the parent `Evaluator.type` field and every version's `type` discriminator — a version's `type` must always match its parent evaluator's `type`. 

## Enum

* `TEMPLATE` (value: `'template'`)

* `CODE` (value: `'code'`)

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


