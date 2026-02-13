# AnnotationConfigsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**annotation_configs** | [**List[AnnotationConfig]**](AnnotationConfig.md) | A list of annotation configs | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.annotation_configs_list200_response import AnnotationConfigsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of AnnotationConfigsList200Response from a JSON string
annotation_configs_list200_response_instance = AnnotationConfigsList200Response.from_json(json)
# print the JSON string representation of the object
print(AnnotationConfigsList200Response.to_json())

# convert the object into a dict
annotation_configs_list200_response_dict = annotation_configs_list200_response_instance.to_dict()
# create an instance of AnnotationConfigsList200Response from a dict
annotation_configs_list200_response_from_dict = AnnotationConfigsList200Response.from_dict(annotation_configs_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


