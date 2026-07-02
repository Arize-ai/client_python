# CategoricalAnnotationConfigUpdate


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | New name for the annotation config. Must be unique within the space. | [optional] 
**annotation_config_type** | **str** | Discriminator value identifying a categorical annotation config. The config &#x60;type&#x60; is immutable and must match the stored config&#39;s type.  | 
**values** | [**List[CategoricalAnnotationValue]**](CategoricalAnnotationValue.md) | The full replacement set of categorical annotation values (2–100 items).  | [optional] 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) | New optimization direction. | [optional] 

## Example

```python
from arize._generated.api_client.models.categorical_annotation_config_update import CategoricalAnnotationConfigUpdate

# TODO update the JSON string below
json = "{}"
# create an instance of CategoricalAnnotationConfigUpdate from a JSON string
categorical_annotation_config_update_instance = CategoricalAnnotationConfigUpdate.from_json(json)
# print the JSON string representation of the object
print(CategoricalAnnotationConfigUpdate.to_json())

# convert the object into a dict
categorical_annotation_config_update_dict = categorical_annotation_config_update_instance.to_dict()
# create an instance of CategoricalAnnotationConfigUpdate from a dict
categorical_annotation_config_update_from_dict = CategoricalAnnotationConfigUpdate.from_dict(categorical_annotation_config_update_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


