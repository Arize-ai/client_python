# VertexAiConfig

Config for a Google Vertex AI integration. Vertex stores no credentials: Arize accesses Vertex through the configured GCP project. All fields are returned on read.

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**provider** | **str** | Discriminator identifying the Vertex AI provider. | 
**project_id** | **str** | GCP project ID Arize accesses Vertex through. | 
**location** | **str** | GCP region (e.g. us-central1). | 
**project_access_label** | **str** | Label used to verify Arize&#39;s access to the GCP project. | 

## Example

```python
from arize._generated.api_client.models.vertex_ai_config import VertexAiConfig

# TODO update the JSON string below
json = "{}"
# create an instance of VertexAiConfig from a JSON string
vertex_ai_config_instance = VertexAiConfig.from_json(json)
# print the JSON string representation of the object
print(VertexAiConfig.to_json())

# convert the object into a dict
vertex_ai_config_dict = vertex_ai_config_instance.to_dict()
# create an instance of VertexAiConfig from a dict
vertex_ai_config_from_dict = VertexAiConfig.from_dict(vertex_ai_config_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


