# ListExperimentsResponse


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**experiments** | [**List[Experiment]**](Experiment.md) | A list of experiments | 
**pagination** | [**PaginationMetadata**](PaginationMetadata.md) |  | 

## Example

```python
from arize._generated.api_client.models.list_experiments_response import ListExperimentsResponse

# TODO update the JSON string below
json = "{}"
# create an instance of ListExperimentsResponse from a JSON string
list_experiments_response_instance = ListExperimentsResponse.from_json(json)
# print the JSON string representation of the object
print(ListExperimentsResponse.to_json())

# convert the object into a dict
list_experiments_response_dict = list_experiments_response_instance.to_dict()
# create an instance of ListExperimentsResponse from a dict
list_experiments_response_from_dict = ListExperimentsResponse.from_dict(list_experiments_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


