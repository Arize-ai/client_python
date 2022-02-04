import datetime
from itertools import chain
from typing import List, Dict, Any, Optional, Union

import pandas as pd
import numpy as np
import pyarrow as pa

import arize.pandas.validation.errors as err
from arize.utils.types import ModelTypes, Environments


class Validator:
    @staticmethod
    def validate_params(
        dataframe: pd.DataFrame,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: "Schema",
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[err.ValidationError]:

        return list(
            chain.from_iterable(
                (
                    Validator._check_invalid_model_id(model_id),
                    Validator._check_invalid_model_version(model_version, environment),
                    Validator._check_invalid_model_type(model_type),
                    Validator._check_invalid_environment(environment),
                    Validator._check_invalid_batch_id(batch_id, environment),
                    Validator._check_existence_pred_act_shap(schema),
                    Validator._check_existence_preprod_pred_act(schema, environment),
                    Validator._check_missing_columns(dataframe, schema),
                )
            )
        )

    @staticmethod
    def validate_types(
        model_type: ModelTypes, schema: "Schema", pyarrow_schema: pa.Schema
    ) -> List[err.ValidationError]:

        column_types = dict(zip(pyarrow_schema.names, pyarrow_schema.types))

        return list(
            chain.from_iterable(
                (
                    Validator._check_type_prediction_id(schema, column_types),
                    Validator._check_type_timestamp(schema, column_types),
                    Validator._check_type_features(schema, column_types),
                    Validator._check_type_shap_values(schema, column_types),
                    Validator._check_type_pred_act_labels(
                        model_type, schema, column_types
                    ),
                    Validator._check_type_pred_act_scores(
                        model_type, schema, column_types
                    ),
                    Validator._check_type_num_seq(model_type, schema, column_types),
                )
            )
        )

    @staticmethod
    def validate_values(
        dataframe: pd.DataFrame, schema: "Schema"
    ) -> List[err.ValidationError]:
        # ASSUMPTION: at this point the param and type checks should have passed.
        # This function may crash if that is not true, e.g. if columns are missing
        # or are of the wrong types.

        return list(
            chain.from_iterable(
                (
                    Validator._check_value_timestamp(dataframe, schema),
                    Validator._check_value_missing(dataframe, schema),
                )
            )
        )

    # ----------------
    # Parameter checks
    # ----------------

    @staticmethod
    def _check_missing_columns(
        dataframe: pd.DataFrame, schema: "Schema"
    ) -> List[err.MissingColumns]:
        # converting to a set first makes the checks run a lot faster
        existing_columns = set(dataframe.columns)
        missing_columns = []

        for field in schema.__dataclass_fields__:
            if field.endswith("column_name"):
                col = getattr(schema, field)
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if schema.feature_column_names is not None:
            for col in schema.feature_column_names:
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if schema.shap_values_column_names is not None:
            for _, col in schema.shap_values_column_names.items():
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if missing_columns:
            return [err.MissingColumns(missing_columns)]
        return []

    @staticmethod
    def _check_invalid_model_id(model_id: Optional[str]) -> List[err.InvalidModelId]:
        # assume it's been coerced to string beforehand
        if (not isinstance(model_id, str)) or len(model_id.strip()) == 0:
            return [err.InvalidModelId()]
        return []

    @staticmethod
    def _check_invalid_model_version(
        model_version: Optional[str], environment: Environments
    ) -> List[err.InvalidModelId]:
        # assume it's been coerced to string beforehand
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            (not isinstance(model_version, str)) or len(model_version.strip()) == 0
        ):
            return [err.InvalidModelVersion()]
        return []

    @staticmethod
    def _check_invalid_batch_id(
        batch_id: Optional[str],
        environment: Environments,
    ) -> List[err.InvalidBatchId]:
        # assume it's been coerced to string beforehand
        if environment in (Environments.VALIDATION,) and (
            (not isinstance(batch_id, str)) or len(batch_id.strip()) == 0
        ):
            return [err.InvalidBatchId()]
        return []

    @staticmethod
    def _check_invalid_model_type(model_type: ModelTypes) -> List[err.InvalidModelType]:
        if model_type in (mt for mt in ModelTypes):
            return []
        return [err.InvalidModelType()]

    @staticmethod
    def _check_invalid_environment(
        environment: Environments,
    ) -> List[err.InvalidEnvironment]:
        if environment in (env for env in Environments):
            return []
        return [err.InvalidEnvironment()]

    @staticmethod
    def _check_existence_pred_act_shap(
        schema: "Schema",
    ) -> List[err.MissingPredActShap]:
        if (
            schema.prediction_label_column_name is not None
            or schema.actual_label_column_name is not None
            or schema.shap_values_column_names is not None
        ):
            return []
        return [err.MissingPredActShap()]

    @staticmethod
    def _check_existence_preprod_pred_act(
        schema: "Schema",
        environment: Environments,
    ) -> List[err.MissingPreprodPredAct]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            schema.prediction_label_column_name is None
            or schema.actual_label_column_name is None
        ):
            return [err.MissingPreprodPredAct()]
        return []

    # -----------
    # Type checks
    # -----------

    @staticmethod
    def _check_type_prediction_id(
        schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        col = schema.prediction_id_column_name
        if col in column_types:
            # should mirror server side
            allowed_datatypes = (
                pa.string(),
                pa.int64(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
            )
            if column_types[col] not in allowed_datatypes:
                return [
                    err.InvalidType("Prediction IDs", expected_types=["str", "int"])
                ]
        return []

    @staticmethod
    def _check_type_timestamp(
        schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        col = schema.timestamp_column_name
        if col in column_types:
            # should mirror server side
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.date32(),
                pa.date64(),
            )
            t = column_types[col]
            if type(t) != pa.TimestampType and t not in allowed_datatypes:
                return [
                    err.InvalidType(
                        "Prediction timestamp",
                        expected_types=["Date", "Timestamp", "int", "float"],
                    )
                ]
        return []

    @staticmethod
    def _check_type_features(
        schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeFeatures]:
        if schema.feature_column_names is not None:
            # should mirror server side
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.string(),
                pa.bool_(),
                pa.int32(),
                pa.float32(),
                pa.int16(),
                pa.int8(),
            )
            mistyped_columns = []
            for col in schema.feature_column_names:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    mistyped_columns.append(col)
            if mistyped_columns:
                return [
                    err.InvalidTypeFeatures(
                        mistyped_columns, expected_types=["float", "int", "bool", "str"]
                    )
                ]
        return []

    @staticmethod
    def _check_type_shap_values(
        schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeShapValues]:
        if schema.shap_values_column_names is not None:
            # should mirror server side
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.float32(),
                pa.int32(),
            )
            mistyped_columns = []
            for _, col in schema.shap_values_column_names.items():
                if col in column_types and column_types[col] not in allowed_datatypes:
                    mistyped_columns.append(col)
            if mistyped_columns:
                return [
                    err.InvalidTypeShapValues(
                        mistyped_columns, expected_types=["float", "int"]
                    )
                ]
        return []

    @staticmethod
    def _check_type_pred_act_labels(
        model_type: ModelTypes, schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        errors = []
        columns = (
            ("Prediction labels", schema.prediction_label_column_name),
            ("Actual labels", schema.actual_label_column_name),
        )
        if model_type == ModelTypes.SCORE_CATEGORICAL:
            # should mirror server side
            allowed_datatypes = (
                pa.string(),
                pa.int64(),
                pa.bool_(),
                pa.float64(),
                pa.int32(),
                pa.float32(),
                pa.int16(),
                pa.int8(),
            )
            for name, col in columns:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    errors.append(
                        err.InvalidType(
                            name, expected_types=["float", "int", "bool", "str"]
                        )
                    )
        elif model_type == ModelTypes.NUMERIC:
            # should mirror server side
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.float32(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
            )
            for name, col in columns:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    errors.append(
                        err.InvalidType(name, expected_types=["float", "int"])
                    )
        elif model_type == ModelTypes.BINARY:
            # should mirror server side
            allowed_datatypes = (pa.bool_(),)
            for name, col in columns:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    errors.append(err.InvalidType(name, expected_types=["bool"]))
        elif model_type == ModelTypes.CATEGORICAL:
            # should mirror server side
            allowed_datatypes = (pa.string(),)
            for name, col in columns:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    errors.append(err.InvalidType(name, expected_types=["str"]))
        return errors

    @staticmethod
    def _check_type_pred_act_scores(
        model_type: ModelTypes, schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        errors = []
        columns = (
            ("Prediction scores", schema.prediction_score_column_name),
            ("Actual scores", schema.actual_score_column_name),
        )
        if model_type == ModelTypes.SCORE_CATEGORICAL:
            # should mirror server side
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.float32(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
            )
            for name, col in columns:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    errors.append(
                        err.InvalidType(name, expected_types=["float", "int"])
                    )
        return errors

    @staticmethod
    def _check_type_num_seq(
        model_type: ModelTypes, schema: "Schema", column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        errors = []
        columns = (
            ("Actual numeric sequence", schema.actual_numeric_sequence_column_name),
        )
        if model_type == ModelTypes.SCORE_CATEGORICAL:
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.float32(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
                pa.null(),
            )
            for name, col in columns:
                if col not in column_types:
                    continue
                if type(column_types[col]) == pa.ListType:
                    type_ = column_types[col].value_type
                    if type_ not in allowed_datatypes:
                        errors.append(
                            err.InvalidType(
                                name + " elements", expected_types=["float", "int"]
                            )
                        )
                elif column_types[col] != pa.null():
                    errors.append(err.InvalidType(name, expected_types=["list"]))
        return errors

    # ------------
    # Value checks
    # ------------

    @staticmethod
    def _check_value_timestamp(
        dataframe: pd.DataFrame, schema: "Schema"
    ) -> List[Union[err.InvalidValueMissingValue, err.InvalidValueTimestamp]]:
        # Due to the timing difference between checking this here and the data finally
        # hitting the same check on server side, there's a some chance for a false
        # result, i.e. the check here suceeeds but the same check on server side fails.
        col = schema.timestamp_column_name
        if col is not None and col in dataframe.columns and len(dataframe):
            # When a timestamp column has Date and NaN, pyarrow will be fine, but
            # pandas min/max will fail due to type incompatibility. So we check for
            # missing value first.
            if dataframe[col].isnull().values.any():
                return [err.InvalidValueMissingValue("Prediction timestamp", "missing")]

            now_t = datetime.datetime.now()
            lbound, ubound = (
                (now_t - datetime.timedelta(days=365)).timestamp(),
                (now_t + datetime.timedelta(days=365)).timestamp(),
            )

            # faster than pyarrow compute
            stats = dataframe[col].agg(["min", "max"])

            ta = pa.Table.from_pandas(stats.to_frame())
            type_ = ta.column(0).type
            min_, max_ = ta.column(0)

            # this part needs improvement: dealing with python types is hard :(
            if (
                (
                    type(type_) == pa.TimestampType
                    and (
                        stats["min"].timestamp() < lbound
                        or stats["max"].timestamp() > ubound
                    )
                )
                or (
                    type_ in (pa.int64(), pa.float64())
                    and (min_.as_py() < lbound or max_.as_py() > ubound)
                )
                or (
                    type_ == pa.date32()
                    and (
                        int(min_.cast(pa.int32()).as_py() * 60 * 60 * 24) < lbound
                        or int(max_.cast(pa.int32()).as_py() * 60 * 60 * 24) > ubound
                    )
                )
                or (
                    type_ == pa.date64()
                    and (
                        int(min_.cast(pa.int64()).as_py() // 1000) < lbound
                        or int(max_.cast(pa.int64()).as_py() // 1000) > ubound
                    )
                )
            ):
                return [
                    err.InvalidValueTimestamp(
                        "Prediction timestamp", acceptible_range="one year"
                    )
                ]

        return []

    @staticmethod
    def _check_value_missing(
        dataframe: pd.DataFrame, schema: "Schema"
    ) -> List[err.InvalidValueMissingValue]:
        errors = []
        columns = (
            ("Prediction IDs", schema.prediction_id_column_name),
            ("Prediction labels", schema.prediction_label_column_name),
            ("Actual labels", schema.actual_label_column_name),
        )
        for name, col in columns:
            if col is not None and col in dataframe.columns:
                if dataframe[col].isnull().any():
                    errors.append(
                        err.InvalidValueMissingValue(name, missingness="missing")
                    )
                elif (
                    dataframe[col].dtype in (np.dtype("float64"), np.dtype("float32"))
                    and np.isinf(dataframe[col]).any()
                ):
                    errors.append(
                        err.InvalidValueMissingValue(name, missingness="infinite")
                    )
        return errors
