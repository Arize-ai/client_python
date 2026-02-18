# ContinuousAnnotationConfigCreate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**annotation_config_type** | **str** | The type of the annotation config | 
**minimum_score** | **float** | The minimum score value | 
**maximum_score** | **float** | The maximum score value | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) |  | [optional] 

## Example

```python
from arize._generated.api_client.models.continuous_annotation_config_create import ContinuousAnnotationConfigCreate

# TODO update the JSON string below
json = "{}"
# create an instance of ContinuousAnnotationConfigCreate from a JSON string
continuous_annotation_config_create_instance = ContinuousAnnotationConfigCreate.from_json(json)
# print the JSON string representation of the object
print(ContinuousAnnotationConfigCreate.to_json())

# convert the object into a dict
continuous_annotation_config_create_dict = continuous_annotation_config_create_instance.to_dict()
# create an instance of ContinuousAnnotationConfigCreate from a dict
continuous_annotation_config_create_from_dict = ContinuousAnnotationConfigCreate.from_dict(continuous_annotation_config_create_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


