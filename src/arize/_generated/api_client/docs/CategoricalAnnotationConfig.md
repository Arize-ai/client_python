# CategoricalAnnotationConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the annotation config | 
**name** | **str** | The name of the annotation config | 
**created_at** | **datetime** | The timestamp for when the annotation config was created | 
**space_id** | **str** | The space id the annotation config belongs to | 
**type** | **str** | The type of the annotation config | 
**values** | [**List[CategoricalAnnotationValue]**](CategoricalAnnotationValue.md) | An array of categorical annotation values | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.categorical_annotation_config import CategoricalAnnotationConfig

# TODO update the JSON string below
json = "{}"
# create an instance of CategoricalAnnotationConfig from a JSON string
categorical_annotation_config_instance = CategoricalAnnotationConfig.from_json(json)
# print the JSON string representation of the object
print(CategoricalAnnotationConfig.to_json())

# convert the object into a dict
categorical_annotation_config_dict = categorical_annotation_config_instance.to_dict()
# create an instance of CategoricalAnnotationConfig from a dict
categorical_annotation_config_from_dict = CategoricalAnnotationConfig.from_dict(categorical_annotation_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


