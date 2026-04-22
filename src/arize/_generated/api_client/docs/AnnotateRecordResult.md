# AnnotateRecordResult

The annotation result for a single annotated record.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_id** | **str** | The ID of the record that was annotated, which is either the dataset example ID or the experiment run ID. | 
**annotations** | [**List[Annotation]**](Annotation.md) | The annotations that were written to this record. | 

## Example

```python
from arize._generated.api_client.models.annotate_record_result import AnnotateRecordResult

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateRecordResult from a JSON string
annotate_record_result_instance = AnnotateRecordResult.from_json(json)
# print the JSON string representation of the object
print(AnnotateRecordResult.to_json())

# convert the object into a dict
annotate_record_result_dict = annotate_record_result_instance.to_dict()
# create an instance of AnnotateRecordResult from a dict
annotate_record_result_from_dict = AnnotateRecordResult.from_dict(annotate_record_result_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


