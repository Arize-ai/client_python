# UpdateContinuousAnnotationConfigRequest


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
from arize._generated.api_client.models.update_continuous_annotation_config_request import UpdateContinuousAnnotationConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of UpdateContinuousAnnotationConfigRequest from a JSON string
update_continuous_annotation_config_request_instance = UpdateContinuousAnnotationConfigRequest.from_json(json)
# print the JSON string representation of the object
print(UpdateContinuousAnnotationConfigRequest.to_json())

# convert the object into a dict
update_continuous_annotation_config_request_dict = update_continuous_annotation_config_request_instance.to_dict()
# create an instance of UpdateContinuousAnnotationConfigRequest from a dict
update_continuous_annotation_config_request_from_dict = UpdateContinuousAnnotationConfigRequest.from_dict(update_continuous_annotation_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


