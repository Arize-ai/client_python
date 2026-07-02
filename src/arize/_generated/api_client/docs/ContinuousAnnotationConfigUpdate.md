# ContinuousAnnotationConfigUpdate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New name for the annotation config. Must be unique within the space. | [optional] 
**annotation_config_type** | **str** | Discriminator value identifying a continuous annotation config. The config &#x60;type&#x60; is immutable and must match the stored config&#39;s type.  | 
**minimum_score** | **float** | New minimum score value. | [optional] 
**maximum_score** | **float** | New maximum score value. | [optional] 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) | New optimization direction. | [optional] 

## Example

```python
from arize._generated.api_client.models.continuous_annotation_config_update import ContinuousAnnotationConfigUpdate

# TODO update the JSON string below
json = "{}"
# create an instance of ContinuousAnnotationConfigUpdate from a JSON string
continuous_annotation_config_update_instance = ContinuousAnnotationConfigUpdate.from_json(json)
# print the JSON string representation of the object
print(ContinuousAnnotationConfigUpdate.to_json())

# convert the object into a dict
continuous_annotation_config_update_dict = continuous_annotation_config_update_instance.to_dict()
# create an instance of ContinuousAnnotationConfigUpdate from a dict
continuous_annotation_config_update_from_dict = ContinuousAnnotationConfigUpdate.from_dict(continuous_annotation_config_update_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


