from itertools import chain
from typing import List, Optional

import pandas as pd

from arize.pandas.tracing.columns import SPAN_OPENINFERENCE_REQUIRED_COLUMNS
from arize.pandas.tracing.validation.common import (
    argument_validation as common_arg_validation,
)
from arize.pandas.tracing.validation.common import (
    dataframe_form_validation as common_df_validation,
)
from arize.pandas.tracing.validation.common import (
    value_validation as common_value_validation,
)
from arize.pandas.tracing.validation.spans import (
    dataframe_form_validation as df_validation,
)
from arize.pandas.tracing.validation.spans import value_validation
from arize.pandas.validation import errors as err


def validate_argument_types(
    spans_dataframe: pd.DataFrame,
    project_name: str,
    dt_fmt: str,
    model_version: Optional[str] = None,
) -> List[err.ValidationError]:
    checks = chain(
        common_arg_validation._check_field_convertible_to_str(
            project_name, model_version
        ),
        common_arg_validation._check_dataframe_type(spans_dataframe),
        common_arg_validation._check_datetime_format_type(dt_fmt),
    )
    return list(checks)


def validate_dataframe_form(
    spans_dataframe: pd.DataFrame,
) -> List[err.ValidationError]:
    df_validation._log_info_dataframe_extra_column_names(spans_dataframe)
    checks = chain(
        # Common
        common_df_validation._check_dataframe_index(spans_dataframe),
        common_df_validation._check_dataframe_required_column_set(
            spans_dataframe,
            required_columns=[
                col.name for col in SPAN_OPENINFERENCE_REQUIRED_COLUMNS
            ],
        ),
        common_df_validation._check_dataframe_for_duplicate_columns(
            spans_dataframe
        ),
        # Spans specific
        df_validation._check_dataframe_column_content_type(spans_dataframe),
    )
    return list(checks)


def validate_values(
    spans_dataframe: pd.DataFrame,
    project_name: str,
    model_version: Optional[str] = None,
) -> List[err.ValidationError]:
    checks = chain(
        # Common
        common_value_validation._check_invalid_project_name(project_name),
        common_value_validation._check_invalid_model_version(model_version),
        # Spans specific
        value_validation._check_span_root_field_values(spans_dataframe),
        value_validation._check_span_attributes_values(spans_dataframe),
    )
    return list(checks)
