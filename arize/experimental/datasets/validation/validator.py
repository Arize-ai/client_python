from typing import List

import pandas as pd

from . import errors as err


class Validator:
    @staticmethod
    def validate(
        df: pd.DataFrame,
    ) -> List[err.DatasetError]:
        ## check all require columns are present
        required_columns_errors = Validator._check_required_columns(df)
        if required_columns_errors:
            return required_columns_errors

        ## check id column is unique
        id_column_unique_constraint_error = Validator._check_id_column_is_unique(df)
        if id_column_unique_constraint_error:
            return id_column_unique_constraint_error

        return []

    @staticmethod
    def _check_required_columns(df: pd.DataFrame) -> List[err.DatasetError]:
        required_columns = ["id", "created_at", "updated_at"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            return [err.RequiredColumnsError(missing_columns)]
        return []

    @staticmethod
    def _check_id_column_is_unique(df: pd.DataFrame) -> List[err.DatasetError]:
        if not df["id"].is_unique:
            return [err.IDColumnUniqueConstraintError]
        return []
