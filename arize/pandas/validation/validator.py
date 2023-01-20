import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa

import arize.pandas.validation.errors as err
from arize.utils.types import Environments, Metrics, ModelTypes, Schema
from arize.utils.utils import MODEL_MAPPING_CONFIG


class Validator:
    @staticmethod
    def validate_required_checks(
        model_id: str,
        schema: Schema,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[err.ValidationError]:

        # minimum required checks
        required_checks = chain(
            Validator._check_field_convertible_to_str(
                model_id, model_version, batch_id
            ),
            Validator._check_field_type_embedding_features_column_names(schema),
        )

        return list(required_checks)

    @staticmethod
    def validate_params(
        dataframe: pd.DataFrame,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: Schema,
        metric_families: Optional[List[Metrics]] = None,
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
            # model mapping checks
            Validator._check_model_type_and_metrics(
                model_type, metric_families, schema
            ),
        )

        if model_type == ModelTypes.NUMERIC:
            num_checks = chain(
                Validator._check_existence_pred_act_shap_score_or_label(schema),
                Validator._check_existence_preprod_pred_act_score_or_label(
                    schema, environment
                ),
            )
            return list(chain(general_checks, num_checks))
        elif model_type == ModelTypes.SCORE_CATEGORICAL:
            sc_checks = chain(
                Validator._check_existence_pred_act_shap(schema),
                Validator._check_existence_preprod_pred_act(schema, environment),
                Validator._check_existence_pred_label(schema),
            )
            return list(chain(general_checks, sc_checks))
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
        )

        if model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_value_rank(dataframe, schema),
                Validator._check_value_prediction_group_id(dataframe, schema),
                Validator._check_value_ranking_category(dataframe, schema),
            )
            return list(chain(general_checks, r_checks))
        return list(general_checks)

    # ----------------------
    # Minimum requred checks
    # ----------------------
    @staticmethod
    def _check_field_convertible_to_str(
        model_id, model_version, batch_id
    ) -> List[err.InvalidFieldTypeConversion]:
        # converting to a set first makes the checks run a lot faster
        wrong_fields = []
        if model_id is not None and not isinstance(model_id, str):
            try:
                str(model_id)
            except:
                wrong_fields.append("model_id")
        if model_version is not None and not isinstance(model_version, str):
            try:
                str(model_version)
            except:
                wrong_fields.append("model_version")
        if batch_id is not None and not isinstance(batch_id, str):
            try:
                str(batch_id)
            except:
                wrong_fields.append("batch_id")

        if wrong_fields:
            return [err.InvalidFieldTypeConversion(wrong_fields, "string")]
        return []

    @staticmethod
    def _check_field_type_embedding_features_column_names(
        schema: Schema,
    ) -> List[err.InvalidFieldTypeEmbeddingFeatures]:
        if schema.embedding_feature_column_names is not None and not isinstance(
            schema.embedding_feature_column_names, dict
        ):
            return [err.InvalidFieldTypeEmbeddingFeatures()]
        return []

    # ----------------
    # Parameter checks
    # ----------------

    @staticmethod
    def _check_model_type_and_metrics(
        model_type: ModelTypes, metric_families: List[Metrics], schema: Schema
    ) -> List[err.ValidationError]:
        if not metric_families:
            return []

        external_model_types = MODEL_MAPPING_CONFIG.get("external_model_types")
        if not external_model_types:
            return []
        if model_type.name.lower() not in external_model_types:
            # model_type is an old model type, e.g. SCORE_CATEGORICAL.
            # We can't do model mapping validations with this type.
            return []

        required_columns_map = MODEL_MAPPING_CONFIG.get("required_columns_map")
        if not required_columns_map:
            return []

        (
            valid_combination,
            missing_columns,
            suggested_model_metric_combinations,
        ) = Validator._check_model_mapping_combinations(
            model_type, metric_families, schema, required_columns_map
        )
        if not valid_combination:
            # Model type + metrics combination is not valid.
            return [
                err.InvalidModelTypeAndMetricsCombination(
                    model_type, metric_families, suggested_model_metric_combinations
                )
            ]
        if missing_columns:
            # For this model type, the schema is missing columns required for the requested metrics.
            return [
                err.MissingRequiredColumnsMetricsValidation(
                    model_type, metric_families, missing_columns
                )
            ]
        return []

    @staticmethod
    def _check_model_mapping_combinations(
        model_type: ModelTypes,
        metric_families: List[Metrics],
        schema: Schema,
        required_columns_map: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str], List[List[str]]]:
        missing_columns = []
        for item in required_columns_map:
            if model_type.name.lower() == item.get("external_model_type"):
                is_valid_combination = False
                metric_combinations = []
                for mapping in item.get("mappings"):
                    # This is a list of lists of metrics.
                    # There may be a few metric combinations that map to the same column enforcement rules.
                    for metrics_list in mapping.get("metrics"):
                        metric_combinations.append(
                            [metric.upper() for metric in metrics_list]
                        )
                        if set(metrics_list) == set(
                            metric_family.name.lower()
                            for metric_family in metric_families
                        ):
                            # This is a valid combination of model type + metrics.
                            # Now validate that required columns are in the schema.
                            is_valid_combination = True
                            # If no prediction values are present, then latent actuals are being logged,
                            # and we can't validate required columns.
                            if (schema.prediction_label_column_name is not None) or (
                                schema.prediction_score_column_name is not None
                            ):
                                # This is a list of lists.
                                # In some cases, either one set of columns OR another set of columns is required.
                                required_columns = (
                                    mapping.get("required_columns")
                                    .get("arrow")
                                    .get("required")
                                )
                                for column_combination in required_columns:
                                    missing_columns = []
                                    if None in {
                                        getattr(schema, column, None)
                                        for column in column_combination
                                    }:
                                        for column in column_combination:
                                            if not getattr(schema, column, None):
                                                missing_columns.append(column)
                                    else:
                                        break
                if not is_valid_combination:
                    return False, [], metric_combinations
        return True, missing_columns, []

    @staticmethod
    def _check_missing_columns(
        dataframe: pd.DataFrame,
        schema: Schema,
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
            for emb_feat_col_names in schema.embedding_feature_column_names.values():
                if emb_feat_col_names.vector_column_name not in existing_columns:
                    missing_columns.append(emb_feat_col_names.vector_column_name)
                if (
                    emb_feat_col_names.data_column_name is not None
                    and emb_feat_col_names.data_column_name not in existing_columns
                ):
                    missing_columns.append(emb_feat_col_names.data_column_name)
                if (
                    emb_feat_col_names.link_to_data_column_name is not None
                    and emb_feat_col_names.link_to_data_column_name
                    not in existing_columns
                ):
                    missing_columns.append(emb_feat_col_names.link_to_data_column_name)

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
    def _check_existence_pred_act_shap_score_or_label(
        schema: Schema,
    ) -> List[err.MissingPredActShap]:
        if (
            (
                schema.prediction_label_column_name is not None
                or schema.prediction_score_column_name is not None
            )
            or (
                schema.actual_label_column_name is not None
                or schema.actual_score_column_name is not None
            )
            or schema.shap_values_column_names is not None
        ):
            return []
        return [err.MissingPredActShapNumeric()]

    @staticmethod
    def _check_existence_pred_label(
        schema: Schema,
    ) -> List[err.MissingPredLabelScoreCategorical]:
        if (
            schema.prediction_score_column_name is not None
            and schema.actual_label_column_name is not None
            and schema.prediction_label_column_name is None
        ):
            return [err.MissingPredLabelScoreCategorical()]
        return []

    @staticmethod
    def _check_existence_preprod_pred_act_score_or_label(
        schema: Schema,
        environment: Environments,
    ) -> List[err.MissingPreprodPredAct]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            (
                schema.prediction_label_column_name is None
                and schema.prediction_score_column_name is None
            )
            or (
                schema.actual_label_column_name is None
                and schema.actual_score_column_name is None
            )
        ):
            return [err.MissingPreprodPredActNumeric()]
        return []

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
            for emb_feat_col_names in schema.embedding_feature_column_names.values():
                # _check_missing_columns() checks that vector columns are present,
                # hence I assume they are here
                col = emb_feat_col_names.vector_column_name
                if (
                    col in column_types
                    and column_types[col] not in allowed_vector_datatypes
                ):
                    mistyped_vector_columns.append(col)

                if emb_feat_col_names.data_column_name:
                    col = emb_feat_col_names.data_column_name
                    if (
                        col in column_types
                        and column_types[col] not in allowed_data_datatypes
                    ):
                        mistyped_data_columns.append(col)

                if emb_feat_col_names.link_to_data_column_name:
                    col = emb_feat_col_names.link_to_data_column_name
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
            ("Relevance scores", schema.relevance_score_column_name),
        )
        if (
            model_type == ModelTypes.SCORE_CATEGORICAL
            or model_type == ModelTypes.RANKING
        ):
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
                if (
                    col is not None
                    and col in column_types
                    and column_types[col] not in allowed_datatypes
                ):
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
        if schema.relevance_labels_column_name is not None:
            col = schema.relevance_labels_column_name
        elif schema.attributions_column_name is not None:
            col = schema.attributions_column_name
        else:
            col = schema.actual_label_column_name
        if col is not None and col in dataframe.columns and len(dataframe):
            if dataframe[col].isnull().values.any():
                # do not attach duplicated missing value error
                # which would be caught by_check_value_missing
                return []
            if dataframe[col].astype(str).str.len().min() == 0:
                return [err.InvalidRankingCategoryValue(col)]
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
            ("Actual labels", schema.actual_label_column_name),
            ("Prediction Group IDs", schema.prediction_group_id_column_name),
            ("Ranks", schema.rank_column_name),
            ("Attributions", schema.attributions_column_name),
            ("Relevance Score", schema.relevance_score_column_name),
            ("Relevance Labels", schema.relevance_labels_column_name),
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
        if schema.relevance_labels_column_name is not None:
            col = schema.relevance_labels_column_name
        elif schema.attributions_column_name is not None:
            col = schema.attributions_column_name
        else:
            col = schema.actual_label_column_name
        if col is not None and col in column_types:
            allowed_datatypes = (pa.list_(pa.string()), pa.string())
            if column_types[col] not in allowed_datatypes:
                return [
                    err.InvalidType(
                        "relevance labels column for ranking models",
                        expected_types=["list of string", "string"],
                    )
                ]
        return []
