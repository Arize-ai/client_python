# DatasetsExamplesInsertRequest


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**examples** | **List[Dict[str, object]]** | Array of examples to append to the dataset version | 

## Example

```python
from arize._generated.api_client.models.datasets_examples_insert_request import DatasetsExamplesInsertRequest

# TODO update the JSON string below
json = "{}"
# create an instance of DatasetsExamplesInsertRequest from a JSON string
datasets_examples_insert_request_instance = DatasetsExamplesInsertRequest.from_json(json)
# print the JSON string representation of the object
print(DatasetsExamplesInsertRequest.to_json())

# convert the object into a dict
datasets_examples_insert_request_dict = datasets_examples_insert_request_instance.to_dict()
# create an instance of DatasetsExamplesInsertRequest from a dict
datasets_examples_insert_request_from_dict = DatasetsExamplesInsertRequest.from_dict(datasets_examples_insert_request_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


