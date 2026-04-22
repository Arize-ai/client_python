# AnnotationBatchResult

Result of a batch annotation operation. Contains one result entry per annotated record.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**results** | [**List[AnnotateRecordResult]**](AnnotateRecordResult.md) | Per-record annotation results, in the same order as the request. | 

## Example

```python
from arize._generated.api_client.models.annotation_batch_result import AnnotationBatchResult

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationBatchResult from a JSON string
annotation_batch_result_instance = AnnotationBatchResult.from_json(json)
# print the JSON string representation of the object
print(AnnotationBatchResult.to_json())

# convert the object into a dict
annotation_batch_result_dict = annotation_batch_result_instance.to_dict()
# create an instance of AnnotationBatchResult from a dict
annotation_batch_result_from_dict = AnnotationBatchResult.from_dict(annotation_batch_result_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


