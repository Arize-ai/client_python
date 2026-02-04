# DatasetsExamplesList200Response


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | [**List[DatasetExample]**](DatasetExample.md) | Array of example objects from the dataset | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.datasets_examples_list200_response import DatasetsExamplesList200Response

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetsExamplesList200Response from a JSON string
datasets_examples_list200_response_instance = DatasetsExamplesList200Response.from_json(json)
# print the JSON string representation of the object
print(DatasetsExamplesList200Response.to_json())

# convert the object into a dict
datasets_examples_list200_response_dict = datasets_examples_list200_response_instance.to_dict()
# create an instance of DatasetsExamplesList200Response from a dict
datasets_examples_list200_response_from_dict = DatasetsExamplesList200Response.from_dict(datasets_examples_list200_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


