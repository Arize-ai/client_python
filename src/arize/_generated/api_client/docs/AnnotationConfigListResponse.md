# AnnotationConfigListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_configs** | [**List[AnnotationConfig]**](AnnotationConfig.md) | A list of annotation configs | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.annotation_config_list_response import AnnotationConfigListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationConfigListResponse from a JSON string
annotation_config_list_response_instance = AnnotationConfigListResponse.from_json(json)
# print the JSON string representation of the object
print(AnnotationConfigListResponse.to_json())

# convert the object into a dict
annotation_config_list_response_dict = annotation_config_list_response_instance.to_dict()
# create an instance of AnnotationConfigListResponse from a dict
annotation_config_list_response_from_dict = AnnotationConfigListResponse.from_dict(annotation_config_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


