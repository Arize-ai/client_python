import datetime
from itertools import chain
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa

import arize.pandas.validation.errors as err
from arize.utils.types import ModelTypes, Environments, Schema


class Validator:
    @staticmethod
    def validate_params(
        dataframe: pd.DataFrame,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: Schema,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[err.ValidationError]:

        # general checks
        general_checks = chain(
            Validator._check_invalid_model_id(model_id),
            Validator._check_invalid_model_version(model_version),
            Validator._check_invalid_model_type(model_type),
            Validator._check_invalid_environment(environment),
            Validator._check_invalid_batch_id(batch_id, environment),
            Validator._check_missing_columns(dataframe, schema),
            Validator._check_invalid_shap_suffix(schema),
        )

        if model_type in (ModelTypes.SCORE_CATEGORICAL, ModelTypes.NUMERIC):
            scn_checks = chain(
                Validator._check_existence_pred_act_shap(schema),
                Validator._check_existence_preprod_pred_act(schema, environment),
            )
            return list(chain(general_checks, scn_checks))
        elif model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_existence_group_id_rank_category_relevance(schema)
            )
            return list(chain(general_checks, r_checks))

        return list(general_checks)

    @staticmethod
    def validate_types(
        model_type: ModelTypes, schema: Schema, pyarrow_schema: pa.Schema
    ) -> List[err.ValidationError]:

        column_types = dict(zip(pyarrow_schema.names, pyarrow_schema.types))
        general_checks = chain(
            Validator._check_type_prediction_id(schema, column_types),
            Validator._check_type_timestamp(schema, column_types),
            Validator._check_type_features(schema, column_types),
            Validator._check_type_embedding_features(schema, column_types),
            Validator._check_type_tags(schema, column_types),
            Validator._check_type_shap_values(schema, column_types),
            Validator._check_type_num_seq(model_type, schema, column_types),
        )

        if model_type in (ModelTypes.SCORE_CATEGORICAL, ModelTypes.NUMERIC):
            scn_checks = chain(
                Validator._check_type_pred_act_labels(model_type, schema, column_types),
                Validator._check_type_pred_act_scores(model_type, schema, column_types),
            )
            return list(chain(general_checks, scn_checks))

        elif model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_type_prediction_group_id(schema, column_types),
                Validator._check_type_rank(schema, column_types),
                Validator._check_type_ranking_category(schema, column_types),
                Validator._check_type_pred_act_scores(model_type, schema, column_types),
            )
            return list(chain(general_checks, r_checks))

        return list(general_checks)

    @staticmethod
    def validate_values(
        dataframe: pd.DataFrame, schema: Schema, model_type: ModelTypes
    ) -> List[err.ValidationError]:
        # ASSUMPTION: at this point the param and type checks should have passed.
        # This function may crash if that is not true, e.g. if columns are missing
        # or are of the wrong types.

        general_checks = chain(
            Validator._check_value_timestamp(dataframe, schema),
            Validator._check_value_missing(dataframe, schema),
            Validator._check_embedding_features_dimensionality(dataframe, schema),
        )

        if model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_value_rank(dataframe, schema),
                Validator._check_value_prediction_group_id(dataframe, schema),
                Validator._check_value_ranking_category(dataframe, schema),
            )
            return list(chain(general_checks, r_checks))
        return list(general_checks)

    # ----------------
    # Parameter checks
    # ----------------

    @staticmethod
    def _check_missing_columns(
        dataframe: pd.DataFrame, schema: Schema
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
                if col not in existing_columns:
                    missing_columns.append(col)

        if schema.embedding_feature_column_names is not None:
            for emb_col_names in schema.embedding_feature_column_names:
                if emb_col_names.vector_column_name not in existing_columns:
                    missing_columns.append(emb_col_names.vector_column_name)
                if (
                    emb_col_names.data_column_name is not None
                    and emb_col_names.data_column_name not in existing_columns
                ):
                    missing_columns.append(emb_col_names.data_column_name)
                if (
                    emb_col_names.link_to_data_column_name is not None
                    and emb_col_names.link_to_data_column_name not in existing_columns
                ):
                    missing_columns.append(emb_col_names.link_to_data_column_name)

        if schema.tag_column_names is not None:
            for col in schema.tag_column_names:
                if col not in existing_columns:
                    missing_columns.append(col)

        if schema.shap_values_column_names is not None:
            for col in schema.shap_values_column_names.values():
                if col not in existing_columns:
                    missing_columns.append(col)

        if missing_columns:
            return [err.MissingColumns(missing_columns)]
        return []

    @staticmethod
    def _check_invalid_shap_suffix(
        schema: Schema,
    ) -> List[err.MissingColumns]:
        invalid_column_names = set()

        if schema.feature_column_names is not None:
            for col in schema.feature_column_names:
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if schema.tag_column_names is not None:
            for col in schema.tag_column_names:
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if schema.shap_values_column_names is not None:
            for col in schema.shap_values_column_names.keys():
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if invalid_column_names:
            return [err.InvalidShapSuffix(invalid_column_names)]
        return []

    @staticmethod
    def _check_invalid_model_id(model_id: Optional[str]) -> List[err.InvalidModelId]:
        # assume it's been coerced to string beforehand
        if (not isinstance(model_id, str)) or len(model_id.strip()) == 0:
            return [err.InvalidModelId()]
        return []

    @staticmethod
    def _check_invalid_model_version(
        model_version: Optional[str] = None,
    ) -> List[err.InvalidModelVersion]:
        if model_version is None:
            return []
        if not isinstance(model_version, str) or len(model_version.strip()) == 0:
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
        schema: Schema,
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
        schema: Schema,
        environment: Environments,
    ) -> List[err.MissingPreprodPredAct]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            schema.prediction_label_column_name is None
            or schema.actual_label_column_name is None
        ):
            return [err.MissingPreprodPredAct()]
        return []

    @staticmethod
    def _check_existence_group_id_rank_category_relevance(
        schema: Schema,
    ) -> List[err.MissingRequiredColumnsForRankingModel]:
        # prediction_group_id and rank columns are required as ranking prediction columns.
        required = (
            schema.prediction_group_id_column_name,
            schema.rank_column_name,
        )
        if any(col is None for col in required):
            return [err.MissingRequiredColumnsForRankingModel()]
        return []

    # -----------
    # Type checks
    # -----------

    @staticmethod
    def _check_type_prediction_id(
        schema: Schema, column_types: Dict[str, Any]
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
        schema: Schema, column_types: Dict[str, Any]
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
        schema: Schema, column_types: Dict[str, Any]
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
    def _check_type_embedding_features(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeFeatures]:
        if schema.embedding_feature_column_names is not None:
            # should mirror server side
            allowed_vector_datatypes = (
                pa.list_(pa.float64()),
                pa.list_(pa.float32()),
            )
            allowed_data_datatypes = (
                pa.string(),  # Text
                pa.list_(pa.string()),  # Token Array
            )
            allowed_link_to_data_datatypes = (pa.string(),)

            mistyped_vector_columns = []
            mistyped_data_columns = []
            mistyped_link_to_data_columns = []
            for embFeatColNames in schema.embedding_feature_column_names:
                # _check_missing_columns() checks that vector columns are present,
                # hence I assume they are here
                col = embFeatColNames.vector_column_name
                if (
                    col in column_types
                    and column_types[col] not in allowed_vector_datatypes
                ):
                    mistyped_vector_columns.append(col)

                if embFeatColNames.data_column_name:
                    col = embFeatColNames.data_column_name
                    if (
                        col in column_types
                        and column_types[col] not in allowed_data_datatypes
                    ):
                        mistyped_data_columns.append(col)

                if embFeatColNames.link_to_data_column_name:
                    col = embFeatColNames.link_to_data_column_name
                    if (
                        col in column_types
                        and column_types[col] not in allowed_link_to_data_datatypes
                    ):
                        mistyped_link_to_data_columns.append(col)

            mistyped_embedding_errors = []
            if mistyped_vector_columns:
                mistyped_embedding_errors.append(
                    err.InvalidTypeFeatures(
                        mistyped_vector_columns,
                        expected_types=["list[float], np.array[float]"],
                    )
                )
            if mistyped_data_columns:
                mistyped_embedding_errors.append(
                    err.InvalidTypeFeatures(
                        mistyped_data_columns, expected_types=["list[string]"]
                    )
                )
            if mistyped_link_to_data_columns:
                mistyped_embedding_errors.append(
                    err.InvalidTypeFeatures(
                        mistyped_link_to_data_columns, expected_types=["string"]
                    )
                )

            return mistyped_embedding_errors  # Will be empty list if no errors

        return []

    @staticmethod
    def _check_type_tags(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeTags]:
        if schema.tag_column_names is not None:
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
            for col in schema.tag_column_names:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    mistyped_columns.append(col)
            if mistyped_columns:
                return [
                    err.InvalidTypeTags(
                        mistyped_columns, ["float", "int", "bool", "str"]
                    )
                ]
        return []

    @staticmethod
    def _check_type_shap_values(
        schema: Schema, column_types: Dict[str, Any]
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
        model_type: ModelTypes, schema: Schema, column_types: Dict[str, Any]
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
        return errors

    @staticmethod
    def _check_type_pred_act_scores(
        model_type: ModelTypes, schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        errors = []
        columns = (
            ("Prediction scores", schema.prediction_score_column_name),
            ("Actual scores", schema.actual_score_column_name),
        )
        if model_type == ModelTypes.SCORE_CATEGORICAL or model_type == ModelTypes.RANKING:
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
        model_type: ModelTypes, schema: Schema, column_types: Dict[str, Any]
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
    def _check_value_rank(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidRankValue]:
        col = schema.rank_column_name
        lbound, ubound = (1, 100)

        if col is not None and col in dataframe.columns and len(dataframe):
            rank_min_max = dataframe[col].agg(["min", "max"])
            if rank_min_max["min"] < lbound or rank_min_max["max"] > ubound:
                return [err.InvalidRankValue(col, "1-100")]
        return []

    @staticmethod
    def _check_value_prediction_group_id(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidPredictionGroupIDLength]:
        col = schema.prediction_group_id_column_name
        if col is not None and col in dataframe.columns and len(dataframe):
            min_len, max_len = (1, 36)
            col_str_lenths = dataframe[col].astype(str).str.len()
            if col_str_lenths.min() < min_len or col_str_lenths.max() > max_len:
                return [
                    err.InvalidPredictionGroupIDLength(
                        name=col, acceptable_range=f"{min_len}-{max_len}"
                    )
                ]
        return []

    @staticmethod
    def _check_value_ranking_category(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[Union[err.InvalidValueMissingValue, err.InvalidRankingCategoryValue]]:
        col = schema.actual_label_column_name
        if col is not None and col in dataframe.columns and len(dataframe):
            if dataframe[col].isnull().values.any():
                # do not attach duplicated missing value error
                # which would be caught by_check_value_missing
                return []
            # empty list
            if dataframe[col].map(len).min() == 0:
                return [err.InvalidValueMissingValue(col, "empty list")]
            # no empty string in list
            if dataframe[col].map(lambda x: (min(map(len, x)))).min() == 0:
                return [err.InvalidRankingCategoryValue(col)]
        return []

    @staticmethod
    def _check_value_timestamp(
        dataframe: pd.DataFrame, schema: Schema
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
                        "Prediction timestamp", acceptable_range="one year"
                    )
                ]

        return []

    @staticmethod
    def _check_value_missing(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidValueMissingValue]:
        errors = []
        columns = (
            ("Prediction IDs", schema.prediction_id_column_name),
            ("Prediction labels", schema.prediction_label_column_name),
            # for ranking models, the ranking:category is specified as actual label
            ("Actual labels", schema.actual_label_column_name),
            ("Prediction Group IDs", schema.prediction_group_id_column_name),
            ("Ranks", schema.rank_column_name),
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

    @staticmethod
    def _check_embedding_features_dimensionality(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.ValidationError]:
        if schema.embedding_feature_column_names is None:
            return []

        multiple_dimensionality_vector_columns = []
        low_dimensionality_vector_columns = []
        for embFeatColNames in schema.embedding_feature_column_names:
            # _check_missing_columns() checks that vector columns are present,
            # hence I assume they are here
            vector_col = embFeatColNames.vector_column_name
            vector_series = dataframe[vector_col]

            if (
                len(vector_series) > 0
                and (vector_series.apply(len) != len(vector_series[0])).any()
            ):
                multiple_dimensionality_vector_columns.append(vector_col)
                continue

            dim = len(vector_series[0])
            if dim <= 1:
                low_dimensionality_vector_columns.append(vector_col)

        wrong_embedding_vector_columns = []
        if multiple_dimensionality_vector_columns:
            wrong_embedding_vector_columns.append(
                err.InvalidValueMultipleEmbeddingVectorDimensionality(
                    multiple_dimensionality_vector_columns
                )
            )
        if low_dimensionality_vector_columns:
            wrong_embedding_vector_columns.append(
                err.InvalidValueLowEmbeddingVectorDimensionality(
                    low_dimensionality_vector_columns
                )
            )
        return wrong_embedding_vector_columns  # Will be empty list if no errors

    @staticmethod
    def _check_type_prediction_group_id(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        col = schema.prediction_group_id_column_name
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
                    err.InvalidType(
                        "prediction_group_ids", expected_types=["str", "int"]
                    )
                ]
        return []

    @staticmethod
    def _check_type_rank(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        col = schema.rank_column_name
        if col in column_types:
            allowed_datatypes = (
                pa.int64(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
            )
            if column_types[col] not in allowed_datatypes:
                return [err.InvalidType("rank", expected_types=["int"])]
        return []

    @staticmethod
    def _check_type_ranking_category(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidType]:
        col = schema.actual_label_column_name
        if col in column_types:
            allowed_datatypes = (pa.list_(pa.string()),)
            if column_types[col] not in allowed_datatypes:
                return [
                    err.InvalidType(
                        "actual label column for ranking models",
                        expected_types=["list of string"],
                    )
                ]
        return []
