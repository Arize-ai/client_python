import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import arize.pandas.validation.errors as err
import numpy as np
import pandas as pd
import pyarrow as pa
from arize.utils.types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
    EmbeddingColumnNames,
    Environments,
    Metrics,
    ModelTypes,
    Schema,
)
from arize.utils.utils import MAX_PREDICTION_ID_LEN, MIN_PREDICTION_ID_LEN, MODEL_MAPPING_CONFIG


class Validator:
    @staticmethod
    def validate_required_checks(
        dataframe: pd.DataFrame,
        model_id: str,
        schema: Schema,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> List[err.ValidationError]:
        # minimum required checks
        required_checks = chain(
            Validator._check_field_convertible_to_str(model_id, model_version, batch_id),
            Validator._check_field_type_embedding_features_column_names(schema),
            Validator._check_field_type_prompt_response(schema),
            Validator._check_invalid_index(dataframe),
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
            Validator._check_model_type_and_metrics(model_type, metric_families, schema),
        )

        if model_type in NUMERIC_MODEL_TYPES:
            num_checks = chain(
                Validator._check_existence_pred_act_shap_score_or_label(schema),
                Validator._check_existence_preprod_pred_act_score_or_label(schema, environment),
                Validator._check_missing_object_detection_columns(schema, model_type),
            )
            return list(chain(general_checks, num_checks))
        elif model_type in CATEGORICAL_MODEL_TYPES:
            sc_checks = chain(
                Validator._check_existence_pred_act_shap(schema),
                Validator._check_existence_preprod_pred_act(schema, environment),
                Validator._check_existence_pred_label(schema),
                Validator._check_missing_object_detection_columns(schema, model_type),
            )
            return list(chain(general_checks, sc_checks))
        elif model_type == ModelTypes.GENERATIVE_LLM:
            gllm_checks = chain(
                Validator._check_existence_prompt_response_generative_llm(schema),
                Validator._check_existence_preprod_act(schema, environment),
                Validator._check_missing_object_detection_columns(schema, model_type),
            )
            return list(chain(general_checks, gllm_checks))
        elif model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_existence_group_id_rank_category_relevance(schema),
                Validator._check_missing_object_detection_columns(schema, model_type),
            )
            return list(chain(general_checks, r_checks))
        elif model_type == ModelTypes.OBJECT_DETECTION:
            od_checks = chain(
                Validator._check_existence_pred_act_od_column_names(schema, environment),
                Validator._check_missing_non_object_detection_columns(schema, model_type),
            )
            return list(chain(general_checks, od_checks))
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
        )

        if model_type in CATEGORICAL_MODEL_TYPES or model_type in NUMERIC_MODEL_TYPES:
            scn_checks = chain(
                Validator._check_type_pred_act_labels(model_type, schema, column_types),
                Validator._check_type_pred_act_scores(model_type, schema, column_types),
            )
            return list(chain(general_checks, scn_checks))
        if model_type == ModelTypes.GENERATIVE_LLM:
            gllm_checks = chain(
                Validator._check_type_pred_act_labels(model_type, schema, column_types),
                Validator._check_type_pred_act_scores(model_type, schema, column_types),
                Validator._check_type_prompt_response(schema, column_types),
            )
            return list(chain(general_checks, gllm_checks))
        elif model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_type_prediction_group_id(schema, column_types),
                Validator._check_type_rank(schema, column_types),
                Validator._check_type_ranking_category(schema, column_types),
                Validator._check_type_pred_act_scores(model_type, schema, column_types),
            )
            return list(chain(general_checks, r_checks))
        elif model_type == ModelTypes.OBJECT_DETECTION:
            od_checks = chain(
                Validator._check_type_bounding_boxes_coordinates(schema, column_types),
                Validator._check_type_bounding_boxes_categories(schema, column_types),
                Validator._check_type_bounding_boxes_scores(schema, column_types),
            )
            return list(chain(general_checks, od_checks))

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
            Validator._check_value_prediction_id(dataframe, schema),
            Validator._check_embedding_features_dimensionality(dataframe, schema),
        )

        if model_type == ModelTypes.RANKING:
            r_checks = chain(
                Validator._check_value_rank(dataframe, schema),
                Validator._check_value_prediction_group_id(dataframe, schema),
                Validator._check_value_ranking_category(dataframe, schema),
            )
            return list(chain(general_checks, r_checks))
        if model_type == ModelTypes.OBJECT_DETECTION:
            od_checks = chain(
                Validator._check_value_bounding_boxes_coordinates(dataframe, schema),
                Validator._check_value_bounding_boxes_categories(dataframe, schema),
                Validator._check_value_bounding_boxes_scores(dataframe, schema),
            )
            return list(chain(general_checks, od_checks))
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
            except Exception:
                wrong_fields.append("model_id")
        if model_version is not None and not isinstance(model_version, str):
            try:
                str(model_version)
            except Exception:
                wrong_fields.append("model_version")
        if batch_id is not None and not isinstance(batch_id, str):
            try:
                str(batch_id)
            except Exception:
                wrong_fields.append("batch_id")

        if wrong_fields:
            return [err.InvalidFieldTypeConversion(wrong_fields, "string")]
        return []

    @staticmethod
    def _check_field_type_embedding_features_column_names(
        schema: Schema,
    ) -> List[err.InvalidFieldTypeEmbeddingFeatures]:
        if schema.embedding_feature_column_names is not None:
            if not isinstance(schema.embedding_feature_column_names, dict):
                return [err.InvalidFieldTypeEmbeddingFeatures()]
            for k, v in schema.embedding_feature_column_names.items():
                if not isinstance(k, str) or not isinstance(v, EmbeddingColumnNames):
                    return [err.InvalidFieldTypeEmbeddingFeatures()]
        return []

    @staticmethod
    def _check_field_type_prompt_response(
        schema: Schema,
    ) -> List[err.InvalidFieldTypePromptResponse]:
        if schema.response_column_names is not None and not isinstance(
            schema.response_column_names, EmbeddingColumnNames
        ):
            return [err.InvalidFieldTypePromptResponse("response_column_names")]
        if schema.prompt_column_names is not None and not isinstance(
            schema.prompt_column_names, EmbeddingColumnNames
        ):
            return [err.InvalidFieldTypePromptResponse("prompt_column_names")]
        return []

    @staticmethod
    def _check_invalid_index(dataframe: pd.DataFrame) -> List[err.InvalidIndex]:
        if (dataframe.index != dataframe.reset_index(drop=True).index).any():
            return [err.InvalidIndex()]
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
                    # There may be a few metric combinations that map to the same column
                    # enforcement rules.
                    for metrics_list in mapping.get("metrics"):
                        metric_combinations.append([metric.upper() for metric in metrics_list])
                        if set(metrics_list) == set(
                            metric_family.name.lower() for metric_family in metric_families
                        ):
                            # This is a valid combination of model type + metrics.
                            # Now validate that required columns are in the schema.
                            is_valid_combination = True
                            # If no prediction values are present, then latent actuals are being
                            # logged, and we can't validate required columns.
                            if (schema.prediction_label_column_name is not None) or (
                                schema.prediction_score_column_name is not None
                            ):
                                # This is a list of lists.
                                # In some cases, either one set of columns OR another set of
                                # columns is required.
                                required_columns = (
                                    mapping.get("required_columns").get("arrow").get("required")
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
                    and emb_feat_col_names.link_to_data_column_name not in existing_columns
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

        if schema.object_detection_prediction_column_names is not None:
            for col in schema.object_detection_prediction_column_names:
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if schema.object_detection_actual_column_names is not None:
            for col in schema.object_detection_actual_column_names:
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if missing_columns:
            return [err.MissingColumns(missing_columns)]
        return []

    @staticmethod
    def _check_invalid_shap_suffix(
        schema: Schema,
    ) -> List[err.InvalidShapSuffix]:
        invalid_column_names = set()

        if schema.feature_column_names is not None:
            for col in schema.feature_column_names:
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if schema.embedding_feature_column_names is not None:
            for emb_col_names in schema.embedding_feature_column_names.values():
                for col in emb_col_names:
                    if col is not None and isinstance(col, str) and col.endswith("_shap"):
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
    def _check_existence_prompt_response_generative_llm(
        schema: Schema,
    ) -> List[err.MissingPromptResponseGenerativeLLM]:
        if schema.prompt_column_names is None or schema.response_column_names is None:
            return [err.MissingPromptResponseGenerativeLLM()]
        return []

    @staticmethod
    def _check_existence_pred_act_shap_score_or_label(
        schema: Schema,
    ) -> List[err.MissingPredActShapNumeric]:
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
    ) -> List[err.MissingPreprodPredActNumeric]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            (
                schema.prediction_label_column_name is None
                and schema.prediction_score_column_name is None
            )
            or (schema.actual_label_column_name is None and schema.actual_score_column_name is None)
        ):
            return [err.MissingPreprodPredActNumeric()]
        return []

    @staticmethod
    def _check_existence_pred_act_od_column_names(
        schema: Schema, environment: Environments
    ) -> List[err.MissingObjectDetectionPredAct]:
        # Checks that the required prediction/actual columns are given in the schema depending on
        # the environment, for object detection models
        if environment == Environments.PRODUCTION:
            if (
                schema.object_detection_prediction_column_names is None
                and schema.object_detection_actual_column_names is None
            ):
                return [err.MissingObjectDetectionPredAct(environment)]
        elif environment in (Environments.TRAINING, Environments.VALIDATION):
            if (
                schema.object_detection_prediction_column_names is None
                or schema.object_detection_actual_column_names is None
            ):
                return [err.MissingObjectDetectionPredAct(environment)]
        return []

    @staticmethod
    def _check_missing_object_detection_columns(
        schema: Schema, model_type: ModelTypes
    ) -> List[err.InvalidPredActObjectDetectionColumnNamesForModelType]:
        # Checks that models that are not Object Detection models don't have, in the schema, the
        # object detection dedicated prediciton/actual column names
        if (
            schema.object_detection_prediction_column_names is not None
            or schema.object_detection_actual_column_names is not None
        ):
            return [err.InvalidPredActObjectDetectionColumnNamesForModelType(model_type)]
        return []

    @staticmethod
    def _check_missing_non_object_detection_columns(
        schema: Schema, model_type: ModelTypes
    ) -> List[err.InvalidPredActColumnNamesForObjectDetectionModelType]:
        # Checks that object detection models don't have, in the schema, the columns reserved for
        # other model types
        columns_to_check = (
            schema.prediction_label_column_name,
            schema.prediction_score_column_name,
            schema.actual_label_column_name,
            schema.actual_score_column_name,
            schema.prediction_group_id_column_name,
            schema.rank_column_name,
            schema.attributions_column_name,
            schema.relevance_score_column_name,
            schema.relevance_labels_column_name,
        )
        wrong_cols = []
        for col in columns_to_check:
            if col is not None:
                wrong_cols.append(col)
        if wrong_cols:
            return [err.InvalidPredActColumnNamesForObjectDetectionModelType(wrong_cols)]
        return []

    @staticmethod
    def _check_existence_preprod_pred_act(
        schema: Schema,
        environment: Environments,
    ) -> List[err.MissingPreprodPredAct]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            schema.prediction_label_column_name is None or schema.actual_label_column_name is None
        ):
            return [err.MissingPreprodPredAct()]
        return []

    @staticmethod
    def _check_existence_preprod_act(
        schema: Schema,
        environment: Environments,
    ) -> List[err.MissingPreprodAct]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            schema.actual_label_column_name is None
        ):
            return [err.MissingPreprodAct()]
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
                return [err.InvalidType("Prediction IDs", expected_types=["str", "int"])]
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
            wrong_type_cols = []
            for col in schema.feature_column_names:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    wrong_type_cols.append(col)
            if wrong_type_cols:
                return [
                    err.InvalidTypeFeatures(
                        wrong_type_cols, expected_types=["float", "int", "bool", "str"]
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

            wrong_type_vector_columns = []
            wrong_type_data_columns = []
            wrong_type_link_to_data_columns = []
            for emb_feat_col_names in schema.embedding_feature_column_names.values():
                # _check_missing_columns() checks that vector columns are present,
                # hence I assume they are here
                col = emb_feat_col_names.vector_column_name
                if col in column_types and column_types[col] not in allowed_vector_datatypes:
                    wrong_type_vector_columns.append(col)

                if emb_feat_col_names.data_column_name:
                    col = emb_feat_col_names.data_column_name
                    if col in column_types and column_types[col] not in allowed_data_datatypes:
                        wrong_type_data_columns.append(col)

                if emb_feat_col_names.link_to_data_column_name:
                    col = emb_feat_col_names.link_to_data_column_name
                    if (
                        col in column_types
                        and column_types[col] not in allowed_link_to_data_datatypes
                    ):
                        wrong_type_link_to_data_columns.append(col)

            wrong_type_embedding_errors = []
            if wrong_type_vector_columns:
                wrong_type_embedding_errors.append(
                    err.InvalidTypeFeatures(
                        wrong_type_vector_columns,
                        expected_types=["list[float], np.array[float]"],
                    )
                )
            if wrong_type_data_columns:
                wrong_type_embedding_errors.append(
                    err.InvalidTypeFeatures(
                        wrong_type_data_columns, expected_types=["list[string]"]
                    )
                )
            if wrong_type_link_to_data_columns:
                wrong_type_embedding_errors.append(
                    err.InvalidTypeFeatures(
                        wrong_type_link_to_data_columns, expected_types=["string"]
                    )
                )

            return wrong_type_embedding_errors  # Will be empty list if no errors

        return []

    @staticmethod
    def _check_type_tags(schema: Schema, column_types: Dict[str, Any]) -> List[err.InvalidTypeTags]:
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
            wrong_type_cols = []
            for col in schema.tag_column_names:
                if col in column_types and column_types[col] not in allowed_datatypes:
                    wrong_type_cols.append(col)
            if wrong_type_cols:
                return [err.InvalidTypeTags(wrong_type_cols, ["float", "int", "bool", "str"])]
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
            wrong_type_cols = []
            for _, col in schema.shap_values_column_names.items():
                if col in column_types and column_types[col] not in allowed_datatypes:
                    wrong_type_cols.append(col)
            if wrong_type_cols:
                return [err.InvalidTypeShapValues(wrong_type_cols, expected_types=["float", "int"])]
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
        if model_type in CATEGORICAL_MODEL_TYPES or model_type == ModelTypes.GENERATIVE_LLM:
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
                        err.InvalidType(name, expected_types=["float", "int", "bool", "str"])
                    )
        elif model_type in NUMERIC_MODEL_TYPES:
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
                    errors.append(err.InvalidType(name, expected_types=["float", "int"]))
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
            model_type in CATEGORICAL_MODEL_TYPES
            or model_type == ModelTypes.RANKING
            or model_type == ModelTypes.GENERATIVE_LLM
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
                    errors.append(err.InvalidType(name, expected_types=["float", "int"]))
        return errors

    @staticmethod
    def _check_type_prompt_response(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidTypePromptResponse]:
        fields_to_check = []
        if schema.prompt_column_names is not None:
            fields_to_check.append(schema.prompt_column_names)
        if schema.response_column_names is not None:
            fields_to_check.append(schema.response_column_names)

        # should mirror server side
        allowed_vector_datatypes = (
            pa.list_(pa.float64()),
            pa.list_(pa.float32()),
        )
        allowed_data_datatypes = (
            pa.string(),  # Text
            pa.list_(pa.string()),  # Token Array
        )
        wrong_type_vector_columns = []
        wrong_type_data_columns = []
        for field in fields_to_check:
            # _check_missing_columns() checks that vector columns are present,
            # hence we assume they are here
            col = field.vector_column_name
            if col in column_types and column_types[col] not in allowed_vector_datatypes:
                wrong_type_vector_columns.append(col)

            if field.data_column_name:
                col = field.data_column_name
                if col in column_types and column_types[col] not in allowed_data_datatypes:
                    wrong_type_data_columns.append(col)

        wrong_type_embedding_errors = []
        if wrong_type_vector_columns:
            wrong_type_embedding_errors.append(
                err.InvalidTypePromptResponse(
                    wrong_type_vector_columns,
                    expected_types=["list[float], np.array[float]"],
                )
            )
        if wrong_type_data_columns:
            wrong_type_embedding_errors.append(
                err.InvalidTypePromptResponse(
                    wrong_type_data_columns, expected_types=["list[string]"]
                )
            )

        return wrong_type_embedding_errors  # Will be empty list if no errors

    @staticmethod
    def _check_type_bounding_boxes_coordinates(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeColumns]:
        # should mirror server side
        allowed_coordinate_types = (
            pa.list_(pa.list_(pa.float64())),
            pa.list_(pa.list_(pa.float32())),
            pa.list_(pa.list_(pa.int64())),
            pa.list_(pa.list_(pa.int32())),
            pa.list_(pa.list_(pa.int16())),
            pa.list_(pa.list_(pa.int8())),
        )
        wrong_type_cols = []

        if schema.object_detection_prediction_column_names is not None:
            coord_col = (
                schema.object_detection_prediction_column_names.bounding_boxes_coordinates_column_name
            )
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

        if schema.object_detection_actual_column_names is not None:
            coord_col = (
                schema.object_detection_actual_column_names.bounding_boxes_coordinates_column_name
            )
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

        return (
            [
                err.InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols,
                    expected_types=["List[List[float]]"],
                )
            ]
            if wrong_type_cols
            else []
        )

    @staticmethod
    def _check_type_bounding_boxes_categories(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeColumns]:
        # should mirror server side
        allowed_category_datatypes = (pa.list_(pa.string()),)  # List of categories
        wrong_type_cols = []

        if schema.object_detection_prediction_column_names is not None:
            cat_col_name = schema.object_detection_prediction_column_names.categories_column_name
            if (
                cat_col_name in column_types
                and column_types[cat_col_name] not in allowed_category_datatypes
            ):
                wrong_type_cols.append(cat_col_name)

        if schema.object_detection_actual_column_names is not None:
            cat_col_name = schema.object_detection_actual_column_names.categories_column_name
            if (
                cat_col_name in column_types
                and column_types[cat_col_name] not in allowed_category_datatypes
            ):
                wrong_type_cols.append(cat_col_name)

        return (
            [
                err.InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols, expected_types=["List[str]"]
                )
            ]
            if wrong_type_cols
            else []
        )

    @staticmethod
    def _check_type_bounding_boxes_scores(
        schema: Schema, column_types: Dict[str, Any]
    ) -> List[err.InvalidTypeColumns]:
        # should mirror server side
        allowed_score_datatypes = (
            pa.list_(pa.float64()),
            pa.list_(pa.float32()),
            pa.list_(pa.int64()),
            pa.list_(pa.int32()),
            pa.list_(pa.int16()),
            pa.list_(pa.int8()),
        )
        wrong_type_cols = []

        if schema.object_detection_prediction_column_names is not None:
            score_col_name = schema.object_detection_prediction_column_names.scores_column_name
            if (
                score_col_name is not None
                and score_col_name in column_types
                and column_types[score_col_name] not in allowed_score_datatypes
            ):
                wrong_type_cols.append(score_col_name)

        if schema.object_detection_actual_column_names is not None:
            score_col_name = schema.object_detection_actual_column_names.scores_column_name
            if (
                score_col_name is not None
                and score_col_name in column_types
                and column_types[score_col_name] not in allowed_score_datatypes
            ):
                wrong_type_cols.append(score_col_name)

        return (
            [
                err.InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols, expected_types=["List[float]"]
                )
            ]
            if wrong_type_cols
            else []
        )

    # ------------
    # Value checks
    # ------------

    @staticmethod
    def _check_embedding_features_dimensionality(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.ValidationError]:
        if schema.embedding_feature_column_names is None:
            return []

        low_dimensionality_vector_columns = []
        for emb_feat_col_names in schema.embedding_feature_column_names.values():
            # _check_missing_columns() checks that vector columns are present,
            # hence I assume they are here
            vector_col = emb_feat_col_names.vector_column_name
            vector_series = dataframe[vector_col]

            if (
                len(vector_series) > 0
                and (
                    vector_series.apply(lambda vec: 0 if vec is None or vec is np.NaN else len(vec))
                    == 1
                ).any()
            ):
                low_dimensionality_vector_columns.append(vector_col)

        return (
            [err.InvalidValueLowEmbeddingVectorDimensionality(low_dimensionality_vector_columns)]
            if low_dimensionality_vector_columns
            else []
        )

    @staticmethod
    def _check_value_rank(dataframe: pd.DataFrame, schema: Schema) -> List[err.InvalidRankValue]:
        col = schema.rank_column_name
        lbound, ubound = (1, 100)

        if col is not None and col in dataframe.columns and len(dataframe):
            rank_min_max = dataframe[col].agg(["min", "max"])
            if rank_min_max["min"] < lbound or rank_min_max["max"] > ubound:
                return [err.InvalidRankValue(col, "1-100")]
        return []

    @staticmethod
    def _check_value_prediction_id(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidStringLength]:
        """
        Require prediction_id to be a string of length 1 - 128 if the model needs embeddings calculations.
        """
        col = schema.prediction_id_column_name
        if schema.embedding_feature_column_names is not None and not Validator._valid_char_limit(
            col, dataframe, MIN_PREDICTION_ID_LEN, MAX_PREDICTION_ID_LEN
        ):
            return [
                err.InvalidStringLength(
                    schema_name="prediction_id_column_name",
                    col_name=col,
                    min_length=MIN_PREDICTION_ID_LEN,
                    max_length=MAX_PREDICTION_ID_LEN,
                )
            ]
        return []

    @staticmethod
    def _check_value_prediction_group_id(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidStringLength]:
        col = schema.prediction_group_id_column_name
        if not Validator._valid_char_limit(
            col, dataframe, MIN_PREDICTION_ID_LEN, MAX_PREDICTION_ID_LEN
        ):
            return [
                err.InvalidStringLength(
                    schema_name="prediction_group_id_column_name",
                    col_name=col,
                    min_length=MIN_PREDICTION_ID_LEN,
                    max_length=MAX_PREDICTION_ID_LEN,
                )
            ]
        return []

    @staticmethod
    def _valid_char_limit(
        col_name: str, dataframe: pd.DataFrame, min_len: int, max_len: int
    ) -> bool:
        if col_name is not None and col_name in dataframe.columns and len(dataframe):
            if not (dataframe[col_name].astype(str).str.len().between(min_len, max_len).all()):
                return False
        return True

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
                    and (stats["min"].timestamp() < lbound or stats["max"].timestamp() > ubound)
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
                    err.InvalidValueTimestamp("Prediction timestamp", acceptable_range="one year")
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
                    errors.append(err.InvalidValueMissingValue(name, wrong_values="missing"))
                elif (
                    dataframe[col].dtype in (np.dtype("float64"), np.dtype("float32"))
                    and np.isinf(dataframe[col]).any()
                ):
                    errors.append(err.InvalidValueMissingValue(name, wrong_values="infinite"))
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
                return [err.InvalidType("prediction_group_ids", expected_types=["str", "int"])]
        return []

    @staticmethod
    def _check_type_rank(schema: Schema, column_types: Dict[str, Any]) -> List[err.InvalidType]:
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

    @staticmethod
    def _check_value_bounding_boxes_coordinates(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidBoundingBoxesCoordinates]:
        errors = []
        if schema.object_detection_prediction_column_names is not None:
            coords_col_name = (
                schema.object_detection_prediction_column_names.bounding_boxes_coordinates_column_name
            )
            error = _check_value_bounding_boxes_coordinates_helper(dataframe[coords_col_name])
            if error is not None:
                errors.append(error)
        if schema.object_detection_actual_column_names is not None:
            coords_col_name = (
                schema.object_detection_actual_column_names.bounding_boxes_coordinates_column_name
            )
            error = _check_value_bounding_boxes_coordinates_helper(dataframe[coords_col_name])
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_bounding_boxes_categories(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidBoundingBoxesCategories]:
        errors = []
        if schema.object_detection_prediction_column_names is not None:
            cat_col_name = schema.object_detection_prediction_column_names.categories_column_name
            error = _check_value_bounding_boxes_categories_helper(dataframe[cat_col_name])
            if error is not None:
                errors.append(error)
        if schema.object_detection_actual_column_names is not None:
            cat_col_name = schema.object_detection_actual_column_names.categories_column_name
            error = _check_value_bounding_boxes_categories_helper(dataframe[cat_col_name])
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_bounding_boxes_scores(
        dataframe: pd.DataFrame, schema: Schema
    ) -> List[err.InvalidBoundingBoxesScores]:
        errors = []
        if schema.object_detection_prediction_column_names is not None:
            sc_col_name = schema.object_detection_prediction_column_names.scores_column_name
            if sc_col_name is not None:
                error = _check_value_bounding_boxes_scores_helper(dataframe[sc_col_name])
                if error is not None:
                    errors.append(error)
        if schema.object_detection_actual_column_names is not None:
            sc_col_name = schema.object_detection_actual_column_names.scores_column_name
            if sc_col_name is not None:
                error = _check_value_bounding_boxes_scores_helper(dataframe[sc_col_name])
                if error is not None:
                    errors.append(error)
        return errors


def _check_value_bounding_boxes_coordinates_helper(
    coordinates_col: pd.Series,
) -> Union[err.InvalidBoundingBoxesCoordinates, None]:
    def check(boxes):
        # We allow for zero boxes. None coordinates list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if boxes is None:
            raise err.InvalidBoundingBoxesCoordinates(reason="none_boxes")
        for box in boxes:
            if box is None or len(box) == 0:
                raise err.InvalidBoundingBoxesCoordinates(reason="none_or_empty_box")
            error = _box_coordinates_wrong_format(box)
            if error is not None:
                raise error

    try:
        coordinates_col.apply(check)
    except err.InvalidBoundingBoxesCoordinates as e:
        return e
    return None


def _box_coordinates_wrong_format(box_coords):
    if (
        # Coordinates should be a collection of 4 floats
        len(box_coords) != 4
        # Coordinates should be positive
        or any(k < 0 for k in box_coords)
        # Coordinates represent the top-left & bottom-right corners of a box: x1 < x2
        or box_coords[0] >= box_coords[2]
        # Coordinates represent the top-left & bottom-right corners of a box: y1 < y2
        or box_coords[1] >= box_coords[3]
    ):
        return err.InvalidBoundingBoxesCoordinates(reason="boxes_coordinates_wrong_format")


def _check_value_bounding_boxes_categories_helper(
    categories_col: pd.Series,
) -> Union[err.InvalidBoundingBoxesCategories, None]:
    def check(categories):
        # We allow for zero boxes. None category list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if categories is None:
            raise err.InvalidBoundingBoxesCategories(reason="none_category_list")
        for category in categories:
            # Allow for empty string category, no None values
            if category is None:
                raise err.InvalidBoundingBoxesCategories(reason="none_category")

    try:
        categories_col.apply(check)
    except err.InvalidBoundingBoxesCategories as e:
        return e
    return None


def _check_value_bounding_boxes_scores_helper(
    scores_col: pd.Series,
) -> Union[err.InvalidBoundingBoxesScores, None]:
    def check(scores):
        # We allow for zero boxes. None confidence score list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if scores is None:
            raise err.InvalidBoundingBoxesScores(reason="none_score_list")
        for score in scores:
            # Confidence scores are between 0 and 1
            if score < 0 or score > 1:
                raise err.InvalidBoundingBoxesScores(reason="scores_out_of_bounds")

    try:
        scores_col.apply(check)
    except err.InvalidBoundingBoxesScores as e:
        return e
    return None
