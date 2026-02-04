# DatasetsList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**datasets** | [**List[Dataset]**](Dataset.md) | A list of datasets | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.datasets_list200_response import DatasetsList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetsList200Response from a JSON string
datasets_list200_response_instance = DatasetsList200Response.from_json(json)
# print the JSON string representation of the object
print(DatasetsList200Response.to_json())

# convert the object into a dict
datasets_list200_response_dict = datasets_list200_response_instance.to_dict()
# create an instance of DatasetsList200Response from a dict
datasets_list200_response_from_dict = DatasetsList200Response.from_dict(datasets_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


