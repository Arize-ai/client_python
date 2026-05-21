# DatasetListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**datasets** | [**List[Dataset]**](Dataset.md) | A list of datasets | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.dataset_list_response import DatasetListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetListResponse from a JSON string
dataset_list_response_instance = DatasetListResponse.from_json(json)
# print the JSON string representation of the object
print(DatasetListResponse.to_json())

# convert the object into a dict
dataset_list_response_dict = dataset_list_response_instance.to_dict()
# create an instance of DatasetListResponse from a dict
dataset_list_response_from_dict = DatasetListResponse.from_dict(dataset_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


