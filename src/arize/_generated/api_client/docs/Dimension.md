# Dimension


## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**category** | [**DimensionCategory**](DimensionCategory.md) | The category of the monitored dimension. | 
**name** | **str** | Name of the monitored field. Omitted when the category has no concrete field name.   | [optional] 

## Example

```python
from arize._generated.api_client.models.dimension import Dimension

# TODO update the JSON string below
json = "{}"
# create an instance of Dimension from a JSON string
dimension_instance = Dimension.from_json(json)
# print the JSON string representation of the object
print(Dimension.to_json())

# convert the object into a dict
dimension_dict = dimension_instance.to_dict()
# create an instance of Dimension from a dict
dimension_from_dict = Dimension.from_dict(dimension_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


