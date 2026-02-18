# CategoricalAnnotationConfigCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**annotation_config_type** | **str** | The type of the annotation config | 
**values** | [**List[CategoricalAnnotationValue]**](CategoricalAnnotationValue.md) | An array of categorical annotation values | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.categorical_annotation_config_create import CategoricalAnnotationConfigCreate

# TODO update the JSON string below
json = "{}"
# create an instance of CategoricalAnnotationConfigCreate from a JSON string
categorical_annotation_config_create_instance = CategoricalAnnotationConfigCreate.from_json(json)
# print the JSON string representation of the object
print(CategoricalAnnotationConfigCreate.to_json())

# convert the object into a dict
categorical_annotation_config_create_dict = categorical_annotation_config_create_instance.to_dict()
# create an instance of CategoricalAnnotationConfigCreate from a dict
categorical_annotation_config_create_from_dict = CategoricalAnnotationConfigCreate.from_dict(categorical_annotation_config_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


