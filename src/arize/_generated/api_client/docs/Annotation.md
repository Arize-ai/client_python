# Annotation

A human annotation on a record.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | The name of the annotation | 
**score** | **float** | Numeric score for the annotation | [optional] 
**label** | **str** | Categorical label for the annotation | [optional] 
**text** | **str** | Free-form text note for the annotation | [optional] 
**updated_at** | **datetime** | Timestamp when the annotation was last updated | [optional] 
**annotator** | [**AnnotatorUser**](AnnotatorUser.md) | The user who made this annotation | [optional] 

## Example

```python
from arize._generated.api_client.models.annotation import Annotation

# TODO update the JSON string below
json = "{}"
# create an instance of Annotation from a JSON string
annotation_instance = Annotation.from_json(json)
# print the JSON string representation of the object
print(Annotation.to_json())

# convert the object into a dict
annotation_dict = annotation_instance.to_dict()
# create an instance of Annotation from a dict
annotation_from_dict = Annotation.from_dict(annotation_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


