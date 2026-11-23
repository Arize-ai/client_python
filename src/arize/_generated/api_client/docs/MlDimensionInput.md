# MlDimensionInput


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**category** | [**MlDimensionCategory**](MlDimensionCategory.md) |  | 
**name** | **str** | Exact, case-sensitive name of the feature, tag, prediction or actual in the project&#39;s schema for the monitored environment, for example &#x60;fico_score&#x60; or &#x60;scorePredictionLabel&#x60;.  | 

## Example

```python
from arize._generated.api_client.models.ml_dimension_input import MlDimensionInput

# TODO update the JSON string below
json = "{}"
# create an instance of MlDimensionInput from a JSON string
ml_dimension_input_instance = MlDimensionInput.from_json(json)
# print the JSON string representation of the object
print(MlDimensionInput.to_json())

# convert the object into a dict
ml_dimension_input_dict = ml_dimension_input_instance.to_dict()
# create an instance of MlDimensionInput from a dict
ml_dimension_input_from_dict = MlDimensionInput.from_dict(ml_dimension_input_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


