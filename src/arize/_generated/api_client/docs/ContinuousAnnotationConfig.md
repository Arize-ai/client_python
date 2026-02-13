# ContinuousAnnotationConfig


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | **str** | The unique identifier for the annotation config | 
**name** | **str** | The name of the annotation config | 
**created_at** | **datetime** | The timestamp for when the annotation config was created | 
**space_id** | **str** | The space id the annotation config belongs to | 
**type** | **str** | The type of the annotation config | 
**minimum_score** | **float** | The minimum score value | 
**maximum_score** | **float** | The maximum score value | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.continuous_annotation_config import ContinuousAnnotationConfig

# TODO update the JSON string below
json = "{}"
# create an instance of ContinuousAnnotationConfig from a JSON string
continuous_annotation_config_instance = ContinuousAnnotationConfig.from_json(json)
# print the JSON string representation of the object
print(ContinuousAnnotationConfig.to_json())

# convert the object into a dict
continuous_annotation_config_dict = continuous_annotation_config_instance.to_dict()
# create an instance of ContinuousAnnotationConfig from a dict
continuous_annotation_config_from_dict = ContinuousAnnotationConfig.from_dict(continuous_annotation_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


