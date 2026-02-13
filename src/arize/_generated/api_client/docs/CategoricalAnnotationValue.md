# CategoricalAnnotationValue


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**label** | **str** | The label value | 
**score** | **float** | A score to associate with the label | [optional] 

## Example

```python
from arize._generated.api_client.models.categorical_annotation_value import CategoricalAnnotationValue

# TODO update the JSON string below
json = "{}"
# create an instance of CategoricalAnnotationValue from a JSON string
categorical_annotation_value_instance = CategoricalAnnotationValue.from_json(json)
# print the JSON string representation of the object
print(CategoricalAnnotationValue.to_json())

# convert the object into a dict
categorical_annotation_value_dict = categorical_annotation_value_instance.to_dict()
# create an instance of CategoricalAnnotationValue from a dict
categorical_annotation_value_from_dict = CategoricalAnnotationValue.from_dict(categorical_annotation_value_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


