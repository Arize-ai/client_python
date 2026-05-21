# DatasetExampleListResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | [**List[DatasetExample]**](DatasetExample.md) | Array of example objects from the dataset | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.dataset_example_list_response import DatasetExampleListResponse

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetExampleListResponse from a JSON string
dataset_example_list_response_instance = DatasetExampleListResponse.from_json(json)
# print the JSON string representation of the object
print(DatasetExampleListResponse.to_json())

# convert the object into a dict
dataset_example_list_response_dict = dataset_example_list_response_instance.to_dict()
# create an instance of DatasetExampleListResponse from a dict
dataset_example_list_response_from_dict = DatasetExampleListResponse.from_dict(dataset_example_list_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


