from typing import List

import pandas as pd

from ....validation.errors import ValidationError


class MetadataArgumentError(ValidationError):
    def __init__(self, message: str, resolution: str) -> None:
        self.message = message
        self.resolution = resolution

    def __repr__(self) -> str:
        return "Metadata_Argument_Error"

    def error_message(self) -> str:
        return f"{self.message} {self.resolution}"


def validate_argument_types(
    metadata_dataframe, project_name
) -> List[ValidationError]:
    """
    Validates the types of arguments passed to update_spans_metadata.

    Args:
        metadata_dataframe: DataFrame with span IDs and patch documents
        project_name: Name of the project

    Returns:
        A list of validation errors, empty if none found
    """
    errors = []

    # Check metadata_dataframe type
    if not isinstance(metadata_dataframe, pd.DataFrame):
        errors.append(
            MetadataArgumentError(
                "metadata_dataframe must be a pandas DataFrame",
                "The metadata_dataframe argument must be a pandas DataFrame.",
            )
        )

    # Check project_name
    if not isinstance(project_name, str) or not project_name.strip():
        errors.append(
            MetadataArgumentError(
                "project_name must be a non-empty string",
                "The project_name argument must be a non-empty string.",
            )
        )

    return errors
