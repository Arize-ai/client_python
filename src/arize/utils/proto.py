"""Protocol buffer schema utilities for tracing data."""

from arize._generated.protocol.rec import public_pb2 as pb2


def get_pb_schema_tracing(
    project_name: str,
) -> pb2.Schema:
    """Create a protobuf schema for LLM tracing data.

    Args:
        project_name: The name of the project/model.

    Returns:
        A configured pb2.Schema object for tracing environment.
    """
    s = pb2.Schema()
    s.constants.model_id = project_name
    s.constants.environment = pb2.Schema.Environment.TRACING
    s.constants.model_type = pb2.Schema.ModelType.GENERATIVE_LLM
    s.arize_spans.SetInParent()
    return s
