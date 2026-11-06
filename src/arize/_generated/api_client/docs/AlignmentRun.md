# AlignmentRun

One alignment score: how closely an evaluator version's labels matched the human annotations over one dataset version, using the column pair and function its task names. A finished run is never rewritten -- re-scoring creates a new run. 

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the alignment run | 
**created_at** | **datetime** | When the alignment run was created | 
**updated_at** | **datetime** | When the alignment run was last modified | 
**alignment_task_id** | **str** | The unique identifier for the alignment task that defined what this run compared | 
**evaluator_version_id** | **str** | The unique identifier for the evaluator version whose labels were scored | 
**dataset_version_id** | **str** | The unique identifier for the dataset version the comparison ran over | 
**predicted_entity** | [**AlignmentPredictedEntity**](AlignmentPredictedEntity.md) | The experiment or dataset version that supplied the evaluator labels. Recorded when the run starts and retained if scoring fails. Null for older runs whose prediction source was not recorded.  | 
**status** | [**TaskRunStatus**](TaskRunStatus.md) | Status of the run. A run that could not be scored is returned with &#x60;FAILED&#x60; and a &#x60;failed_reason&#x60;; it is still a successfully created run, not a request error.  | 
**score** | **float** | The fraction of comparable rows on which the evaluator&#39;s label and the human annotation agreed, from 0 to 1.  &#x60;null&#x60; means no row had both a judge label and a human annotation, so there was nothing to compare. &#x60;null&#x60; is not the same as &#x60;0&#x60;: &#x60;0&#x60; means rows were compared and none of them agreed. Treating &#x60;null&#x60; as &#x60;0&#x60; reports total disagreement where nothing was measured at all.  Always &#x60;null&#x60; when &#x60;status&#x60; is &#x60;FAILED&#x60;.  | 
**comparable_row_count** | **int** | How many rows had both a judge label and a human annotation. This is the denominator &#x60;score&#x60; was computed over, so a high &#x60;score&#x60; over a small &#x60;comparable_row_count&#x60; is not meaningful -- read the two together.  &#x60;null&#x60; when the count is unknown, which includes every &#x60;FAILED&#x60; run. A value of &#x60;0&#x60; pairs with a &#x60;null&#x60; &#x60;score&#x60;.  | 
**failed_reason** | **str** | Why the run failed, when &#x60;status&#x60; is &#x60;FAILED&#x60;. &#x60;null&#x60; otherwise. | 

## Example

```python
from arize._generated.api_client.models.alignment_run import AlignmentRun

# TODO update the JSON string below
json = "{}"
# create an instance of AlignmentRun from a JSON string
alignment_run_instance = AlignmentRun.from_json(json)
# print the JSON string representation of the object
print(AlignmentRun.to_json())

# convert the object into a dict
alignment_run_dict = alignment_run_instance.to_dict()
# create an instance of AlignmentRun from a dict
alignment_run_from_dict = AlignmentRun.from_dict(alignment_run_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


