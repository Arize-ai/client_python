# CategoricalAnnotationValueRequest

A categorical annotation value in a write request (strict form of CategoricalAnnotationValue).

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**label** | **str** | The label value | 
**score** | **float** | A score to associate with the label | [optional] 

## Example

```python
from arize._generated.api_client.models.categorical_annotation_value_request import CategoricalAnnotationValueRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CategoricalAnnotationValueRequest from a JSON string
categorical_annotation_value_request_instance = CategoricalAnnotationValueRequest.from_json(json)
# print the JSON string representation of the object
print(CategoricalAnnotationValueRequest.to_json())

# convert the object into a dict
categorical_annotation_value_request_dict = categorical_annotation_value_request_instance.to_dict()
# create an instance of CategoricalAnnotationValueRequest from a dict
categorical_annotation_value_request_from_dict = CategoricalAnnotationValueRequest.from_dict(categorical_annotation_value_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


