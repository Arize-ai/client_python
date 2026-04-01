# AnnotationInput

An annotation value to set on a record, identified by its annotation config name. Omitting a field leaves the existing value unchanged.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The annotation config name | 
**score** | **float** | Numeric score for the annotation. Omit to leave unchanged. | [optional] 
**label** | **str** | Categorical label for the annotation. Omit to leave unchanged. | [optional] 
**text** | **str** | Free-form text note for the annotation. Omit to leave unchanged. | [optional] 

## Example

```python
from arize._generated.api_client.models.annotation_input import AnnotationInput

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationInput from a JSON string
annotation_input_instance = AnnotationInput.from_json(json)
# print the JSON string representation of the object
print(AnnotationInput.to_json())

# convert the object into a dict
annotation_input_dict = annotation_input_instance.to_dict()
# create an instance of AnnotationInput from a dict
annotation_input_from_dict = AnnotationInput.from_dict(annotation_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


