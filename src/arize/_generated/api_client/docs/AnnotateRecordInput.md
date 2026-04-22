# AnnotateRecordInput

A single record to annotate in a batch, identified by its record ID.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**record_id** | **str** | The ID of the record to annotate (dataset example ID or experiment run ID). | 
**values** | [**List[AnnotationInput]**](AnnotationInput.md) | One or more annotation values to set on this record. | 

## Example

```python
from arize._generated.api_client.models.annotate_record_input import AnnotateRecordInput

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotateRecordInput from a JSON string
annotate_record_input_instance = AnnotateRecordInput.from_json(json)
# print the JSON string representation of the object
print(AnnotateRecordInput.to_json())

# convert the object into a dict
annotate_record_input_dict = annotate_record_input_instance.to_dict()
# create an instance of AnnotateRecordInput from a dict
annotate_record_input_from_dict = AnnotateRecordInput.from_dict(annotate_record_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


