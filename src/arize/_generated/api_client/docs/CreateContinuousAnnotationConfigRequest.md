# CreateContinuousAnnotationConfigRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **str** | Name of the new annotation config | 
**space_id** | **str** | ID of the space the annotation config will belong to | 
**annotation_config_type** | **str** | Discriminator value identifying a continuous annotation config. | 
**minimum_score** | **float** | The minimum score value | 
**maximum_score** | **float** | The maximum score value | 
**optimization_direction** | [**OptimizationDirection**](OptimizationDirection.md) | Direction for optimization. Defaults to &#x60;NONE&#x60; when omitted. | [optional] 

## Example

```python
from arize._generated.api_client.models.create_continuous_annotation_config_request import CreateContinuousAnnotationConfigRequest

# TODO update the JSON string below
json = "{}"
# create an instance of CreateContinuousAnnotationConfigRequest from a JSON string
create_continuous_annotation_config_request_instance = CreateContinuousAnnotationConfigRequest.from_json(json)
# print the JSON string representation of the object
print(CreateContinuousAnnotationConfigRequest.to_json())

# convert the object into a dict
create_continuous_annotation_config_request_dict = create_continuous_annotation_config_request_instance.to_dict()
# create an instance of CreateContinuousAnnotationConfigRequest from a dict
create_continuous_annotation_config_request_from_dict = CreateContinuousAnnotationConfigRequest.from_dict(create_continuous_annotation_config_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


