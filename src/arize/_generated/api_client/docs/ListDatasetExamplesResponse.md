# ListDatasetExamplesResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | [**List[DatasetExample]**](DatasetExample.md) | Array of example objects from the dataset | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_dataset_examples_response import ListDatasetExamplesResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListDatasetExamplesResponse from a JSON string
list_dataset_examples_response_instance = ListDatasetExamplesResponse.from_json(json)
# print the JSON string representation of the object
print(ListDatasetExamplesResponse.to_json())

# convert the object into a dict
list_dataset_examples_response_dict = list_dataset_examples_response_instance.to_dict()
# create an instance of ListDatasetExamplesResponse from a dict
list_dataset_examples_response_from_dict = ListDatasetExamplesResponse.from_dict(list_dataset_examples_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


