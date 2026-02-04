"""Batch validation logic for ML model predictions and actuals."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from arize.constants.ml import (
    LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME,
    LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME,
    LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME,
    MAX_DOCUMENT_ID_LEN,
    MAX_EMBEDDING_DIMENSIONALITY,
    MAX_FUTURE_YEARS_FROM_CURRENT_TIME,
    MAX_LLM_MODEL_NAME_LENGTH,
    MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION,
    MAX_MULTI_CLASS_NAME_LENGTH,
    MAX_NUMBER_OF_EMBEDDINGS,
    MAX_NUMBER_OF_MULTI_CLASS_CLASSES,
    MAX_PAST_YEARS_FROM_CURRENT_TIME,
    MAX_PREDICTION_ID_LEN,
    MAX_PROMPT_TEMPLATE_LENGTH,
    MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION,
    MAX_PROMPT_TEMPLATE_VERSION_LENGTH,
    MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION,
    MAX_RAW_DATA_CHARACTERS,
    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
    MAX_TAG_LENGTH,
    MAX_TAG_LENGTH_TRUNCATION,
    MIN_DOCUMENT_ID_LEN,
    MIN_PREDICTION_ID_LEN,
    MODEL_MAPPING_CONFIG,
)
from arize.exceptions.base import (
    InvalidDataFrameIndex,
    InvalidFieldTypeConversion,
    ValidationError,
)
from arize.exceptions.types import (
    InvalidFieldTypeLlmConfig,
    InvalidFieldTypePromptTemplates,
    InvalidType,
    InvalidTypeColumns,
    InvalidTypeFeatures,
    InvalidTypeShapValues,
    InvalidTypeTags,
    InvalidValueEmbeddingRawDataTooLong,
    InvalidValueEmbeddingVectorDimensionality,
)
from arize.exceptions.values import (
    InvalidBoundingBoxesCategories,
    InvalidBoundingBoxesCoordinates,
    InvalidBoundingBoxesScores,
    InvalidMultiClassActScoreValue,
    InvalidMultiClassClassNameLength,
    InvalidMultiClassPredScoreValue,
    InvalidMultiClassThresholdClasses,
    InvalidNumClassesMultiClassMap,
    InvalidPolygonCategories,
    InvalidPolygonCoordinates,
    InvalidPolygonScores,
    InvalidRankingCategoryValue,
    InvalidRankValue,
    InvalidRecord,
    InvalidStringLengthInColumn,
    InvalidTagLength,
    InvalidValueMissingValue,
    InvalidValueTimestamp,
)
from arize.logging import get_truncation_warning_message
from arize.ml.batch_validation.errors import (
    DuplicateColumnsInDataframe,
    InvalidBatchId,
    InvalidColumnNameEmptyString,
    InvalidEnvironment,
    InvalidFieldTypeEmbeddingFeatures,
    InvalidFieldTypePromptResponse,
    InvalidModelId,
    InvalidModelType,
    InvalidModelTypeAndMetricsCombination,
    InvalidModelVersion,
    InvalidNumberOfEmbeddings,
    InvalidPredActColumnNamesForModelType,
    InvalidPredActCVColumnNamesForModelType,
    InvalidSchemaType,
    InvalidShapSuffix,
    MissingColumns,
    MissingCVPredAct,
    MissingPredictionIdColumnForDelayedRecords,
    MissingPreprodAct,
    MissingPreprodPredActNumericAndCategorical,
    MissingReqPredActColumnNamesForMultiClass,
    MissingRequiredColumnsForRankingModel,
    MissingRequiredColumnsMetricsValidation,
    MultipleCVPredAct,
    ReservedColumns,
)
from arize.ml.types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
    BaseSchema,
    CorpusSchema,
    EmbeddingColumnNames,
    Environments,
    LLMConfigColumnNames,
    Metrics,
    ModelTypes,
    PromptTemplateColumnNames,
    Schema,
    _normalize_column_names,
    segments_intersect,
)
from arize.utils.types import (
    is_dict_of,
    is_iterable_of,
)

logger = logging.getLogger(__name__)


class Validator:
    """Validator for batch data with schema and dataframe validation methods."""

    @staticmethod
    def validate_required_checks(
        dataframe: pd.DataFrame,
        model_id: str,
        environment: Environments,
        schema: BaseSchema,
        model_version: str | None = None,
        batch_id: str | None = None,
    ) -> list[ValidationError]:
        """Validate required checks for schema, environment, and :class:`pandas.DataFrame` structure."""
        general_checks = chain(
            Validator._check_valid_schema_type(schema, environment),
            Validator._check_field_convertible_to_str(
                model_id, model_version, batch_id
            ),
            Validator._check_invalid_index(dataframe),
        )
        # If the schema is a CorpusSchema then for the log to be valid the environment must
        # be CORPUS. By including both conditions here we do not need to modify the
        # other checks below to account for possible CorpusSchema (which would not be valid
        # in those checks).
        if environment == Environments.CORPUS or isinstance(
            schema, CorpusSchema
        ):
            return list(general_checks)
        if isinstance(schema, Schema):
            return list(
                chain(
                    general_checks,
                    Validator._check_field_type_embedding_features_column_names(
                        schema
                    ),
                    Validator._check_field_type_prompt_response(schema),
                    Validator._check_field_type_prompt_templates(schema),
                    Validator._check_field_type_llm_config(dataframe, schema),
                )
            )
        return []

    @staticmethod
    def validate_params(
        dataframe: pd.DataFrame,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments,
        schema: BaseSchema,
        metric_families: list[Metrics] | None = None,
        model_version: str | None = None,
        batch_id: str | None = None,
    ) -> list[ValidationError]:
        """Validate parameters including model type, environment, and schema consistency."""
        # general checks
        general_checks = chain(
            Validator._check_column_names_for_empty_strings(schema),
            Validator._check_invalid_model_id(model_id),
            Validator._check_invalid_model_version(model_version),
            Validator._check_invalid_model_type(model_type),
            Validator._check_invalid_environment(environment),
            Validator._check_dataframe_for_duplicate_columns(schema, dataframe),
            Validator._check_missing_columns(dataframe, schema),
            Validator._check_reserved_columns(schema, model_type),
        )
        if isinstance(schema, CorpusSchema):
            return list(general_checks)
        if isinstance(schema, Schema):
            general_checks = chain(
                general_checks,
                Validator._check_existence_prediction_id_column_delayed_schema(
                    schema, model_type
                ),
                Validator._check_invalid_batch_id(batch_id, environment),
                Validator._check_invalid_number_of_embeddings(schema),
                Validator._check_invalid_shap_suffix(schema),
                # model mapping checks
                Validator._check_model_type_and_metrics(
                    model_type, metric_families, schema
                ),
            )

            if model_type in NUMERIC_MODEL_TYPES:
                num_checks = chain(
                    Validator._check_existence_preprod_pred_act_score_or_label(
                        schema, environment
                    ),
                    Validator._check_missing_object_detection_columns(
                        schema, model_type
                    ),
                    Validator._check_missing_multi_class_columns(
                        schema, model_type
                    ),
                )
                return list(chain(general_checks, num_checks))
            if model_type in CATEGORICAL_MODEL_TYPES:
                sc_checks = chain(
                    Validator._check_existence_preprod_pred_act_score_or_label(
                        schema, environment
                    ),
                    Validator._check_missing_object_detection_columns(
                        schema, model_type
                    ),
                    Validator._check_missing_multi_class_columns(
                        schema, model_type
                    ),
                )
                return list(chain(general_checks, sc_checks))
            if model_type == ModelTypes.GENERATIVE_LLM:
                gllm_checks = chain(
                    Validator._check_existence_preprod_act(schema, environment),
                    Validator._check_missing_object_detection_columns(
                        schema, model_type
                    ),
                    Validator._check_missing_multi_class_columns(
                        schema, model_type
                    ),
                )
                return list(chain(general_checks, gllm_checks))
            if model_type == ModelTypes.RANKING:
                r_checks = chain(
                    Validator._check_existence_group_id_rank_category_relevance(
                        schema
                    ),
                    Validator._check_missing_object_detection_columns(
                        schema, model_type
                    ),
                    Validator._check_missing_multi_class_columns(
                        schema, model_type
                    ),
                )
                return list(chain(general_checks, r_checks))
            if model_type == ModelTypes.OBJECT_DETECTION:
                od_checks = chain(
                    Validator._check_exactly_one_cv_column_type(
                        schema, environment
                    ),
                    Validator._check_missing_non_object_detection_columns(
                        schema, model_type
                    ),
                    Validator._check_missing_multi_class_columns(
                        schema, model_type
                    ),
                )
                return list(chain(general_checks, od_checks))
            if model_type == ModelTypes.MULTI_CLASS:
                multi_class_checks = chain(
                    Validator._check_existing_multi_class_columns(schema),
                    Validator._check_missing_non_multi_class_columns(
                        schema, ModelTypes.MULTI_CLASS
                    ),
                )
                return list(chain(general_checks, multi_class_checks))
        return list(general_checks)

    @staticmethod
    def validate_types(
        model_type: ModelTypes,
        schema: BaseSchema,
        pyarrow_schema: pa.Schema,
    ) -> list[ValidationError]:
        """Validate column data types against expected types for the schema."""
        column_types = dict(
            zip(pyarrow_schema.names, pyarrow_schema.types, strict=True)
        )

        if isinstance(schema, CorpusSchema):
            return list(
                chain(
                    Validator._check_type_document_columns(schema, column_types)
                )
            )
        if isinstance(schema, Schema):
            general_checks = chain(
                Validator._check_type_prediction_id(schema, column_types),
                Validator._check_type_timestamp(schema, column_types),
                Validator._check_type_features(schema, column_types),
                Validator._check_type_embedding_features(schema, column_types),
                Validator._check_type_tags(schema, column_types),
                Validator._check_type_shap_values(schema, column_types),
                Validator._check_type_retrieved_document_ids(
                    schema, column_types
                ),
            )

            if (
                model_type in CATEGORICAL_MODEL_TYPES
                or model_type in NUMERIC_MODEL_TYPES
            ):
                scn_checks = chain(
                    Validator._check_type_pred_act_labels(
                        model_type, schema, column_types
                    ),
                    Validator._check_type_pred_act_scores(
                        model_type, schema, column_types
                    ),
                )
                return list(chain(general_checks, scn_checks))
            if model_type == ModelTypes.GENERATIVE_LLM:
                gllm_checks = chain(
                    Validator._check_type_pred_act_labels(
                        model_type, schema, column_types
                    ),
                    Validator._check_type_pred_act_scores(
                        model_type, schema, column_types
                    ),
                    Validator._check_type_prompt_response(schema, column_types),
                    Validator._check_type_llm_prompt_templates(
                        schema, column_types
                    ),
                    Validator._check_type_llm_config(schema, column_types),
                    Validator._check_type_llm_run_metadata(
                        schema, column_types
                    ),
                )
                return list(chain(general_checks, gllm_checks))
            if model_type == ModelTypes.RANKING:
                r_checks = chain(
                    Validator._check_type_prediction_group_id(
                        schema, column_types
                    ),
                    Validator._check_type_rank(schema, column_types),
                    Validator._check_type_ranking_category(
                        schema, column_types
                    ),
                    Validator._check_type_pred_act_scores(
                        model_type, schema, column_types
                    ),
                )
                return list(chain(general_checks, r_checks))
            if model_type == ModelTypes.OBJECT_DETECTION:
                od_checks = chain(
                    Validator._check_type_image_segment_coordinates(
                        schema, column_types
                    ),
                    Validator._check_type_image_segment_categories(
                        schema, column_types
                    ),
                    Validator._check_type_image_segment_scores(
                        schema, column_types
                    ),
                )
                return list(chain(general_checks, od_checks))
            if model_type == ModelTypes.MULTI_CLASS:
                multi_class_checks = chain(
                    Validator._check_type_multi_class_pred_threshold_act_scores(
                        schema, column_types
                    ),
                )
                return list(chain(general_checks, multi_class_checks))

            return list(general_checks)
        return []

    @staticmethod
    def validate_values(
        dataframe: pd.DataFrame,
        environment: Environments,
        schema: BaseSchema,
        model_type: ModelTypes,
    ) -> list[ValidationError]:
        """Validate data values including ranges, formats, and consistency checks."""
        # ASSUMPTION: at this point the param and type checks should have passed.
        # This function may crash if that is not true, e.g. if columns are missing
        # or are of the wrong types.
        if len(dataframe) == 0:
            return []

        general_checks = chain(
            Validator._check_invalid_missing_values(
                dataframe, schema, model_type
            ),
        )
        if isinstance(schema, CorpusSchema):
            return list(
                chain(
                    general_checks,
                    Validator._check_document_id_field_str_length(
                        dataframe,
                        "document_id_column_name",
                        schema.document_id_column_name,
                    ),
                )
            )
        if isinstance(schema, Schema):
            general_checks = chain(
                general_checks,
                Validator._check_value_timestamp(dataframe, schema),  # type: ignore[arg-type]
                Validator._check_id_field_str_length(  # type: ignore[arg-type]
                    dataframe,
                    "prediction_id_column_name",
                    schema.prediction_id_column_name,
                ),
                Validator._check_embedding_vectors_dimensionality(  # type: ignore[arg-type]
                    dataframe, schema
                ),
                Validator._check_embedding_raw_data_characters(  # type: ignore[arg-type]
                    dataframe, schema
                ),
                Validator._check_invalid_record_prod(  # type: ignore[arg-type]
                    dataframe, environment, schema, model_type
                ),
                Validator._check_invalid_record_preprod(  # type: ignore[arg-type]
                    dataframe, environment, schema, model_type
                ),
                Validator._check_value_tag(dataframe, schema),  # type: ignore[arg-type]
            )
            if model_type == ModelTypes.RANKING:
                r_checks = chain(
                    Validator._check_value_rank(dataframe, schema),
                    Validator._check_id_field_str_length(
                        dataframe,
                        "prediction_group_id_column_name",
                        schema.prediction_group_id_column_name,
                    ),
                    Validator._check_value_ranking_category(dataframe, schema),
                )
                return list(chain(general_checks, r_checks))
            if model_type == ModelTypes.OBJECT_DETECTION:
                od_checks = chain(
                    Validator._check_value_bounding_boxes_coordinates(
                        dataframe, schema
                    ),
                    Validator._check_value_bounding_boxes_categories(
                        dataframe, schema
                    ),
                    Validator._check_value_bounding_boxes_scores(
                        dataframe, schema
                    ),
                    Validator._check_value_semantic_segmentation_polygon_coordinates(
                        dataframe, schema
                    ),
                    Validator._check_value_semantic_segmentation_polygon_categories(
                        dataframe, schema
                    ),
                    Validator._check_value_instance_segmentation_polygon_categories(
                        dataframe, schema
                    ),
                    Validator._check_value_instance_segmentation_polygon_coordinates(
                        dataframe, schema
                    ),
                    Validator._check_value_instance_segmentation_polygon_scores(
                        dataframe, schema
                    ),
                    Validator._check_value_instance_segmentation_bbox_coordinates(
                        dataframe, schema
                    ),
                )
                return list(chain(general_checks, od_checks))
            if model_type == ModelTypes.GENERATIVE_LLM:
                gen_llm_checks = chain(
                    Validator._check_value_prompt_response(dataframe, schema),
                    Validator._check_value_llm_model_name(dataframe, schema),
                    Validator._check_value_llm_prompt_template(
                        dataframe, schema
                    ),
                    Validator._check_value_llm_prompt_template_version(
                        dataframe, schema
                    ),
                )
                return list(chain(general_checks, gen_llm_checks))
            if model_type == ModelTypes.MULTI_CLASS:
                multi_class_checks = chain(
                    Validator._check_length_multi_class_maps(dataframe, schema),
                    Validator._check_classes_and_scores_values_in_multi_class_maps(
                        dataframe, schema
                    ),
                    Validator._check_each_multi_class_pred_has_threshold(
                        dataframe, schema
                    ),
                )
                return list(chain(general_checks, multi_class_checks))
            return list(general_checks)
        return []

    # -----------------------
    # Minimum required checks
    # -----------------------
    @staticmethod
    def _check_column_names_for_empty_strings(
        schema: BaseSchema,
    ) -> list[InvalidColumnNameEmptyString]:
        if "" in schema.get_used_columns():
            return [InvalidColumnNameEmptyString()]
        return []

    @staticmethod
    def _check_field_convertible_to_str(
        model_id: object, model_version: object, batch_id: object
    ) -> list[InvalidFieldTypeConversion]:
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
            return [InvalidFieldTypeConversion(wrong_fields, "string")]
        return []

    @staticmethod
    def _check_field_type_embedding_features_column_names(
        schema: Schema,
    ) -> list[InvalidFieldTypeEmbeddingFeatures]:
        if schema.embedding_feature_column_names is not None:
            if not isinstance(schema.embedding_feature_column_names, dict):
                return [InvalidFieldTypeEmbeddingFeatures()]
            for k, v in schema.embedding_feature_column_names.items():
                if not isinstance(k, str) or not isinstance(
                    v, EmbeddingColumnNames
                ):
                    return [InvalidFieldTypeEmbeddingFeatures()]
        return []

    @staticmethod
    def _check_field_type_prompt_response(
        schema: Schema,
    ) -> list[InvalidFieldTypePromptResponse]:
        errors: list[InvalidFieldTypePromptResponse] = []
        if schema.prompt_column_names is not None and not isinstance(
            schema.prompt_column_names, (str, EmbeddingColumnNames)
        ):
            errors.append(InvalidFieldTypePromptResponse("prompt_column_names"))
        if schema.response_column_names is not None and not isinstance(
            schema.response_column_names, (str, EmbeddingColumnNames)
        ):
            errors.append(
                InvalidFieldTypePromptResponse("response_column_names")
            )
        return errors

    @staticmethod
    def _check_field_type_prompt_templates(
        schema: Schema,
    ) -> list[InvalidFieldTypePromptTemplates]:
        if schema.prompt_template_column_names is not None and not isinstance(
            schema.prompt_template_column_names, PromptTemplateColumnNames
        ):
            return [InvalidFieldTypePromptTemplates()]
        return []

    @staticmethod
    def _check_field_type_llm_config(
        dataframe: pd.DataFrame,
        schema: Schema,
    ) -> list[InvalidFieldTypeLlmConfig | InvalidTypeColumns]:
        if schema.llm_config_column_names is None:
            return []
        if not isinstance(schema.llm_config_column_names, LLMConfigColumnNames):
            return [InvalidFieldTypeLlmConfig()]
        col = schema.llm_config_column_names.params_column_name
        # We check the types if the columns are in the dataframe.
        # If the columns are reflected in the schema but not present
        # in the dataframe, it will be caught by _check_missing_columns
        if (
            col is not None
            and col in dataframe.columns
            and any(
                not is_dict_of(
                    val,
                    key_allowed_types=str,
                    value_allowed_types=(bool, int, float, str),
                    value_list_allowed_types=str,
                )
                for val in dataframe[col]
            )
        ):
            return [
                InvalidTypeColumns(
                    wrong_type_columns=[col],
                    expected_types=[
                        "Dict[str, (bool, int, float, string or list[str])]"
                    ],
                )
            ]
        return []

    @staticmethod
    def _check_invalid_index(
        dataframe: pd.DataFrame,
    ) -> list[InvalidDataFrameIndex]:
        if (dataframe.index != dataframe.reset_index(drop=True).index).any():
            return [InvalidDataFrameIndex()]
        return []

    # ----------------
    # Parameter checks
    # ----------------

    @staticmethod
    def _check_model_type_and_metrics(
        model_type: ModelTypes,
        metric_families: list[Metrics] | None,
        schema: Schema,
    ) -> list[ValidationError]:
        if metric_families is None:
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
                InvalidModelTypeAndMetricsCombination(
                    model_type,
                    metric_families,
                    suggested_model_metric_combinations,
                )
            ]
        if missing_columns:
            # For this model type, the schema is missing columns required for the requested metrics.
            return [
                MissingRequiredColumnsMetricsValidation(
                    model_type, metric_families, missing_columns
                )
            ]
        return []

    @staticmethod
    def _check_model_mapping_combinations(
        model_type: ModelTypes,
        metric_families: list[Metrics],
        schema: Schema,
        required_columns_map: list[dict[str, Any]],
    ) -> tuple[bool, list[str], list[list[str]]]:
        missing_columns: list[str] = []
        for item in required_columns_map:
            if model_type.name.lower() == item.get("external_model_type"):
                is_valid_combination = False
                metric_combinations = []
                mappings = item.get("mappings")
                if mappings is not None:
                    for mapping in mappings:
                        # This is a list of lists of metrics.
                        # There may be a few metric combinations that map to the same column
                        # enforcement rules.
                        for metrics_list in mapping.get("metrics"):
                            metric_combinations.append(
                                [metric.upper() for metric in metrics_list]
                            )
                            if set(metrics_list) == {
                                metric_family.name.lower()
                                for metric_family in metric_families
                            }:
                                # This is a valid combination of model type + metrics.
                                # Now validate that required columns are in the schema.
                                is_valid_combination = True
                                # If no prediction values are present, then delayed actuals are being
                                # logged, and we can't validate required columns.
                                if schema.has_prediction_columns():
                                    # This is a list of lists.
                                    # In some cases, either one set of columns OR another set of
                                    # columns is required.
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
                                                if not getattr(
                                                    schema, column, None
                                                ):
                                                    missing_columns.append(
                                                        column
                                                    )
                                        else:
                                            break
                if not is_valid_combination:
                    return False, [], metric_combinations
        return True, missing_columns, []

    @staticmethod
    def _check_existence_prediction_id_column_delayed_schema(
        schema: Schema, model_type: ModelTypes
    ) -> list[MissingPredictionIdColumnForDelayedRecords]:
        if schema.prediction_id_column_name is not None:
            return []
        # TODO: Revise logic once prediction_label column addition (for generative models)
        # is moved to beginning of log function
        if schema.is_delayed() and model_type is not ModelTypes.GENERATIVE_LLM:
            # We skip GENERATIVE model types since they are assigned a default
            # prediction label column with values equal 1
            return [
                MissingPredictionIdColumnForDelayedRecords(
                    schema.has_actual_columns(),
                    schema.has_feature_importance_columns(),
                )
            ]
        # We don't allow delayed actuals for generative models, since we give a default prediction
        # label column with 1 as default value
        if model_type is not ModelTypes.GENERATIVE_LLM:
            # Warning for when prediction_id is not provided by the user and we generate the default
            # prediction ids
            logger.warning(
                "Prediction ID is not specified. Arize generates UUIDs for the model's predictions "
                "if not provided by the user. Please note, you won't be able to send delayed data for "
                "joining if a Prediction ID is not provided."
            )

        return []

    @staticmethod
    def _check_missing_columns(
        dataframe: pd.DataFrame,
        schema: BaseSchema,
    ) -> list[MissingColumns]:
        if isinstance(schema, CorpusSchema):
            return Validator._check_missing_columns_corpus_schema(
                dataframe, schema
            )
        if isinstance(schema, Schema):
            return Validator._check_missing_columns_schema(dataframe, schema)
        return []

    @staticmethod
    def _check_missing_columns_schema(
        dataframe: pd.DataFrame,
        schema: Schema,
    ) -> list[MissingColumns]:
        # converting to a set first makes the checks run a lot faster
        existing_columns = set(dataframe.columns)
        missing_columns = []

        for field in schema.__dataclass_fields__:
            if field.endswith("column_name"):
                col = getattr(schema, field)
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if schema.feature_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in _normalize_column_names(
                        schema.feature_column_names
                    )
                    if col not in existing_columns
                ]
            )

        if schema.embedding_feature_column_names is not None:
            for (
                emb_feat_col_names
            ) in schema.embedding_feature_column_names.values():
                if (
                    emb_feat_col_names.vector_column_name
                    not in existing_columns
                ):
                    missing_columns.append(
                        emb_feat_col_names.vector_column_name
                    )
                if (
                    emb_feat_col_names.data_column_name is not None
                    and emb_feat_col_names.data_column_name
                    not in existing_columns
                ):
                    missing_columns.append(emb_feat_col_names.data_column_name)
                if (
                    emb_feat_col_names.link_to_data_column_name is not None
                    and emb_feat_col_names.link_to_data_column_name
                    not in existing_columns
                ):
                    missing_columns.append(
                        emb_feat_col_names.link_to_data_column_name
                    )

        if schema.tag_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in _normalize_column_names(schema.tag_column_names)
                    if col not in existing_columns
                ]
            )

        if schema.shap_values_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.shap_values_column_names.values()
                    if col not in existing_columns
                ]
            )

        if schema.object_detection_prediction_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.object_detection_prediction_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.object_detection_actual_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.object_detection_actual_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.semantic_segmentation_prediction_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.semantic_segmentation_prediction_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.semantic_segmentation_actual_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.semantic_segmentation_actual_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.instance_segmentation_prediction_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.instance_segmentation_prediction_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.instance_segmentation_actual_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.instance_segmentation_actual_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.prompt_column_names is not None:
            if isinstance(schema.prompt_column_names, str):
                col = schema.prompt_column_names
                if col not in existing_columns:
                    missing_columns.append(col)
            elif isinstance(schema.prompt_column_names, EmbeddingColumnNames):
                prompt_emb_col_names = schema.prompt_column_names
                if (
                    prompt_emb_col_names.vector_column_name
                    not in existing_columns
                ):
                    missing_columns.append(
                        prompt_emb_col_names.vector_column_name
                    )
                if (
                    prompt_emb_col_names.data_column_name is not None
                    and prompt_emb_col_names.data_column_name
                    not in existing_columns
                ):
                    missing_columns.append(
                        prompt_emb_col_names.data_column_name
                    )

        if schema.response_column_names is not None:
            if isinstance(schema.response_column_names, str):
                col = schema.response_column_names
                if col not in existing_columns:
                    missing_columns.append(col)
            elif isinstance(schema.response_column_names, EmbeddingColumnNames):
                response_emb_col_names = schema.response_column_names
                if (
                    response_emb_col_names.vector_column_name
                    not in existing_columns
                ):
                    missing_columns.append(
                        response_emb_col_names.vector_column_name
                    )
                if (
                    response_emb_col_names.data_column_name is not None
                    and response_emb_col_names.data_column_name
                    not in existing_columns
                ):
                    missing_columns.append(
                        response_emb_col_names.data_column_name
                    )

        if schema.prompt_template_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.prompt_template_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if schema.llm_config_column_names is not None:
            missing_columns.extend(
                [
                    col
                    for col in schema.llm_config_column_names
                    if col is not None and col not in existing_columns
                ]
            )

        if missing_columns:
            return [MissingColumns(missing_columns)]
        return []

    @staticmethod
    def _check_missing_columns_corpus_schema(
        dataframe: pd.DataFrame,
        schema: CorpusSchema,
    ) -> list[MissingColumns]:
        # converting to a set first makes the checks run a lot faster
        existing_columns = set(dataframe.columns)
        missing_columns = []

        for field in schema.__dataclass_fields__:
            if field.endswith("column_name"):
                col = getattr(schema, field)
                if col is not None and col not in existing_columns:
                    missing_columns.append(col)

        if (
            schema.document_id_column_name is not None
            and schema.document_id_column_name not in existing_columns
        ):
            missing_columns.append(schema.document_id_column_name)
        if (
            schema.document_version_column_name is not None
            and schema.document_version_column_name not in existing_columns
        ):
            missing_columns.append(schema.document_version_column_name)
        if schema.document_text_embedding_column_names is not None:
            if (
                schema.document_text_embedding_column_names.vector_column_name
                is not None
                and schema.document_text_embedding_column_names.vector_column_name
                not in existing_columns
            ):
                missing_columns.append(
                    schema.document_text_embedding_column_names.vector_column_name
                )
            if (
                schema.document_text_embedding_column_names.data_column_name
                is not None
                and schema.document_text_embedding_column_names.data_column_name
                not in existing_columns
            ):
                missing_columns.append(
                    schema.document_text_embedding_column_names.data_column_name
                )
            if (
                schema.document_text_embedding_column_names.link_to_data_column_name
                is not None
                and schema.document_text_embedding_column_names.link_to_data_column_name
                not in existing_columns
            ):
                missing_columns.append(
                    schema.document_text_embedding_column_names.link_to_data_column_name
                )
        if missing_columns:
            return [MissingColumns(missing_columns)]
        return []

    @staticmethod
    def _check_valid_schema_type(
        schema: BaseSchema,
        environment: Environments,
    ) -> list[InvalidSchemaType]:
        if environment == Environments.CORPUS and not (
            isinstance(schema, CorpusSchema)
        ):
            return [
                InvalidSchemaType(
                    schema_type=str(type(schema)), environment=environment
                )
            ]
        if environment != Environments.CORPUS and isinstance(
            schema, CorpusSchema
        ):
            return [
                InvalidSchemaType(
                    schema_type=str(type(schema)), environment=environment
                )
            ]
        return []

    @staticmethod
    def _check_invalid_shap_suffix(
        schema: Schema,
    ) -> list[InvalidShapSuffix]:
        invalid_column_names = set()

        if schema.feature_column_names is not None:
            for col in _normalize_column_names(schema.feature_column_names):
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if schema.embedding_feature_column_names is not None:
            for emb_col_names in schema.embedding_feature_column_names.values():
                cols_list = [c for c in emb_col_names if c is not None]
                for col in cols_list:
                    if col.endswith("_shap"):
                        invalid_column_names.add(col)

        if schema.tag_column_names is not None:
            for col in _normalize_column_names(schema.tag_column_names):
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if schema.shap_values_column_names is not None:
            for col in schema.shap_values_column_names:
                if isinstance(col, str) and col.endswith("_shap"):
                    invalid_column_names.add(col)

        if invalid_column_names:
            return [InvalidShapSuffix(invalid_column_names)]
        return []

    @staticmethod
    def _check_reserved_columns(
        schema: BaseSchema,
        model_type: ModelTypes,
    ) -> list[ReservedColumns]:
        if isinstance(schema, CorpusSchema):
            return []
        if isinstance(schema, Schema):
            reserved_columns = []
            column_counts = schema.get_used_columns_counts()
            if model_type == ModelTypes.GENERATIVE_LLM:
                # Check whether the reserved columns are found in any parts of the schema they are not
                # permitted to be. To do this, count the number of times the reserved columns appear in
                # the schema. If it's found just once, make sure it's in the correct place. If it's found
                # more than once, we know it is somewhere it should not be.
                if (
                    column_counts.get(
                        LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME, 0
                    )
                    == 1
                ) and (
                    not schema.llm_run_metadata_column_names
                    or schema.llm_run_metadata_column_names.total_token_count_column_name
                    != LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME
                    )
                if (
                    column_counts.get(
                        LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME, 0
                    )
                    > 1
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME
                    )

                if (
                    column_counts.get(
                        LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME, 0
                    )
                    == 1
                ) and (
                    not schema.llm_run_metadata_column_names
                    or schema.llm_run_metadata_column_names.prompt_token_count_column_name
                    != LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME
                    )
                if (
                    column_counts.get(
                        LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME, 0
                    )
                    > 1
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME
                    )

                if (
                    column_counts.get(
                        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME, 0
                    )
                    == 1
                ) and (
                    not schema.llm_run_metadata_column_names
                    or schema.llm_run_metadata_column_names.response_token_count_column_name
                    != LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                    )
                if (
                    column_counts.get(
                        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME, 0
                    )
                    > 1
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME
                    )

                if (
                    column_counts.get(
                        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME, 0
                    )
                    == 1
                ) and (
                    not schema.llm_run_metadata_column_names
                    or schema.llm_run_metadata_column_names.response_latency_ms_column_name
                    != LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME
                    )
                if (
                    column_counts.get(
                        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME, 0
                    )
                    > 1
                ):
                    reserved_columns.append(
                        LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME
                    )

            if reserved_columns:
                return [ReservedColumns(reserved_columns)]
        return []

    @staticmethod
    def _check_invalid_model_id(
        model_id: str | None,
    ) -> list[InvalidModelId]:
        # assume it's been coerced to string beforehand
        if (not isinstance(model_id, str)) or len(model_id.strip()) == 0:
            return [InvalidModelId()]
        return []

    @staticmethod
    def _check_invalid_model_version(
        model_version: str | None = None,
    ) -> list[InvalidModelVersion]:
        if model_version is None:
            return []
        if (
            not isinstance(model_version, str)
            or len(model_version.strip()) == 0
        ):
            return [InvalidModelVersion()]

        return []

    @staticmethod
    def _check_invalid_batch_id(
        batch_id: str | None,
        environment: Environments,
    ) -> list[InvalidBatchId]:
        # assume it's been coerced to string beforehand
        if environment in (Environments.VALIDATION,) and (
            (not isinstance(batch_id, str)) or len(batch_id.strip()) == 0
        ):
            return [InvalidBatchId()]
        return []

    @staticmethod
    def _check_invalid_model_type(
        model_type: ModelTypes,
    ) -> list[InvalidModelType]:
        if model_type in (mt for mt in ModelTypes):
            return []
        return [InvalidModelType()]

    @staticmethod
    def _check_invalid_environment(
        environment: Environments,
    ) -> list[InvalidEnvironment]:
        if environment in (env for env in Environments):
            return []
        return [InvalidEnvironment()]

    @staticmethod
    def _check_existence_preprod_pred_act_score_or_label(
        schema: Schema,
        environment: Environments,
    ) -> list[MissingPreprodPredActNumericAndCategorical]:
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
            return [MissingPreprodPredActNumericAndCategorical()]
        return []

    @staticmethod
    def _check_exactly_one_cv_column_type(
        schema: Schema, environment: Environments
    ) -> list[MultipleCVPredAct | MissingCVPredAct]:
        # Checks that the required prediction/actual columns are given in the schema depending on
        # the environment, for object detection models. There should be exactly one of
        # object detection, semantic segmentation, or instance segmentation columns.
        if environment in (Environments.PRODUCTION,):
            # Check how many CV column types are present (should be exactly 1). For production,
            # we only need one of the prediction/actual columns to be present.
            object_detection_columns_present = (
                schema.object_detection_prediction_column_names is not None
                or schema.object_detection_actual_column_names is not None
            )
            semantic_segmentation_columns_present = (
                schema.semantic_segmentation_prediction_column_names is not None
                or schema.semantic_segmentation_actual_column_names is not None
            )
            instance_segmentation_columns_present = (
                schema.instance_segmentation_prediction_column_names is not None
                or schema.instance_segmentation_actual_column_names is not None
            )

            # Count how many types are present
            cv_types_count = sum(
                [
                    object_detection_columns_present,
                    semantic_segmentation_columns_present,
                    instance_segmentation_columns_present,
                ]
            )

            if cv_types_count == 0:
                return [MissingCVPredAct(environment)]
            if cv_types_count > 1:
                return [MultipleCVPredAct(environment)]

        elif environment in (
            Environments.TRAINING,
            Environments.VALIDATION,
        ):
            # Check how many CV column types are present (should be exactly 1). For training and
            # validation, we need both the prediction and actual columns to be present.
            object_detection_columns_present = (
                schema.object_detection_prediction_column_names is not None
                and schema.object_detection_actual_column_names is not None
            )
            semantic_segmentation_columns_present = (
                schema.semantic_segmentation_prediction_column_names is not None
                and schema.semantic_segmentation_actual_column_names is not None
            )
            instance_segmentation_columns_present = (
                schema.instance_segmentation_prediction_column_names is not None
                and schema.instance_segmentation_actual_column_names is not None
            )

            # Count how many types are present
            cv_types_count = sum(
                [
                    object_detection_columns_present,
                    semantic_segmentation_columns_present,
                    instance_segmentation_columns_present,
                ]
            )

            if cv_types_count == 0:
                return [MissingCVPredAct(environment)]
            if cv_types_count > 1:
                return [MultipleCVPredAct(environment)]

        return []

    @staticmethod
    def _check_missing_object_detection_columns(
        schema: Schema, model_type: ModelTypes
    ) -> list[InvalidPredActCVColumnNamesForModelType]:
        # Checks that models that are not Object Detection models don't have, in the schema, the
        # object detection, semantic segmentation, or instance segmentation dedicated prediction/actual
        # column names
        if (
            schema.object_detection_prediction_column_names is not None
            or schema.object_detection_actual_column_names is not None
            or schema.semantic_segmentation_prediction_column_names is not None
            or schema.semantic_segmentation_actual_column_names is not None
            or schema.instance_segmentation_prediction_column_names is not None
            or schema.instance_segmentation_actual_column_names is not None
        ):
            return [InvalidPredActCVColumnNamesForModelType(model_type)]
        return []

    @staticmethod
    def _check_missing_non_object_detection_columns(
        schema: Schema, model_type: ModelTypes
    ) -> list[InvalidPredActColumnNamesForModelType]:
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
        wrong_cols = [col for col in columns_to_check if col is not None]
        if wrong_cols:
            allowed_cols = [
                "object_detection_prediction_column_names",
                "object_detection_actual_column_names",
                "semantic_segmentation_prediction_column_names",
                "semantic_segmentation_actual_column_names",
                "instance_segmentation_prediction_column_names",
                "instance_segmentation_actual_column_names",
            ]
            return [
                InvalidPredActColumnNamesForModelType(
                    model_type, allowed_cols, wrong_cols
                )
            ]
        return []

    @staticmethod
    def _check_missing_multi_class_columns(
        schema: Schema, model_type: ModelTypes
    ) -> list[InvalidPredActColumnNamesForModelType]:
        # Checks that models that are not Multi Class models don't have, in the schema, the
        # multi class dedicated threshold column
        if (
            model_type != ModelTypes.MULTI_CLASS
            and schema.multi_class_threshold_scores_column_name is not None
        ):
            return [
                InvalidPredActColumnNamesForModelType(
                    model_type,
                    None,  # type: ignore[arg-type]
                    [schema.multi_class_threshold_scores_column_name],
                )
            ]
        return []

    @staticmethod
    def _check_existing_multi_class_columns(
        schema: Schema,
    ) -> list[MissingReqPredActColumnNamesForMultiClass]:
        # Checks that models that are Multi Class models have, in the schema, the
        # required prediction score or actual score columns
        if (
            schema.prediction_score_column_name is None
            and schema.actual_score_column_name is None
        ) or (
            schema.multi_class_threshold_scores_column_name is not None
            and schema.prediction_score_column_name is None
        ):
            return [MissingReqPredActColumnNamesForMultiClass()]
        return []

    @staticmethod
    def _check_missing_non_multi_class_columns(
        schema: Schema, model_type: ModelTypes
    ) -> list[InvalidPredActColumnNamesForModelType]:
        # Checks that multi class models don't have, in the schema, the columns reserved for
        # other model types
        columns_to_check = (
            schema.prediction_label_column_name,
            schema.actual_label_column_name,
            schema.prediction_group_id_column_name,
            schema.rank_column_name,
            schema.attributions_column_name,
            schema.relevance_score_column_name,
            schema.relevance_labels_column_name,
            schema.object_detection_prediction_column_names,
            schema.object_detection_actual_column_names,
            schema.semantic_segmentation_prediction_column_names,
            schema.semantic_segmentation_actual_column_names,
            schema.instance_segmentation_prediction_column_names,
            schema.instance_segmentation_actual_column_names,
        )
        wrong_cols = [col for col in columns_to_check if col is not None]
        if wrong_cols:
            allowed_cols = [
                "prediction_score_column_name",
                "multi_class_threshold_scores_column_name",
                "actual_score_column_name",
            ]
            return [
                InvalidPredActColumnNamesForModelType(
                    model_type,
                    allowed_cols,
                    wrong_cols,  # type: ignore[arg-type]
                )
            ]
        return []

    @staticmethod
    def _check_existence_preprod_act(
        schema: Schema,
        environment: Environments,
    ) -> list[MissingPreprodAct]:
        if environment in (Environments.VALIDATION, Environments.TRAINING) and (
            schema.actual_label_column_name is None
        ):
            return [MissingPreprodAct()]
        return []

    @staticmethod
    def _check_existence_group_id_rank_category_relevance(
        schema: Schema,
    ) -> list[MissingRequiredColumnsForRankingModel]:
        # prediction_group_id and rank columns are required as ranking prediction columns.
        ranking_prediction_cols = (
            schema.prediction_label_column_name,
            schema.prediction_score_column_name,
            schema.rank_column_name,
            schema.prediction_group_id_column_name,
        )
        has_prediction_info = any(
            col is not None for col in ranking_prediction_cols
        )
        required = (
            schema.prediction_group_id_column_name,
            schema.rank_column_name,
        )
        # If there is prediction information (not delayed actuals),
        # there must exist a rank and prediction group id columns
        if has_prediction_info and any(col is None for col in required):
            return [MissingRequiredColumnsForRankingModel()]
        return []

    @staticmethod
    def _check_dataframe_for_duplicate_columns(
        schema: BaseSchema, dataframe: pd.DataFrame
    ) -> list[DuplicateColumnsInDataframe]:
        # Get the columns used in the schema
        schema_col_used = schema.get_used_columns()
        # Get the duplicated column names from the dataframe
        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()]
        # These are the duplicated columns from the dataframe that are referred to by the schema
        schema_duplicate_cols = [
            col for col in duplicate_columns if col in schema_col_used
        ]
        if schema_duplicate_cols:
            return [DuplicateColumnsInDataframe(schema_duplicate_cols)]
        return []

    @staticmethod
    def _check_invalid_number_of_embeddings(
        schema: Schema,
    ) -> list[InvalidNumberOfEmbeddings]:
        if schema.embedding_feature_column_names is not None:
            number_of_embeddings = len(schema.embedding_feature_column_names)
            if number_of_embeddings > MAX_NUMBER_OF_EMBEDDINGS:
                return [InvalidNumberOfEmbeddings(number_of_embeddings)]
        return []

    # -----------
    # Type checks
    # -----------

    @staticmethod
    def _check_type_prediction_id(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
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
                    InvalidType(
                        "Prediction IDs",
                        expected_types=["str", "int"],
                        found_data_type=column_types[col],
                    )
                ]
        return []

    @staticmethod
    def _check_type_timestamp(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
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
            if (
                not isinstance(t, pa.TimestampType)
                and t not in allowed_datatypes
            ):
                return [
                    InvalidType(
                        "Prediction timestamp",
                        expected_types=["Date", "Timestamp", "int", "float"],
                        found_data_type=t,
                    )
                ]
        return []

    @staticmethod
    def _check_type_features(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeFeatures]:
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
                pa.null(),
                pa.list_(pa.string()),
            )
            wrong_type_cols = [
                col
                for col in _normalize_column_names(schema.feature_column_names)
                if col in column_types
                and column_types[col] not in allowed_datatypes
            ]
            if wrong_type_cols:
                return [
                    InvalidTypeFeatures(
                        wrong_type_cols,
                        expected_types=[
                            "float",
                            "int",
                            "bool",
                            "str",
                            "list[str]",
                        ],
                    )
                ]
        return []

    @staticmethod
    def _check_type_embedding_features(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeFeatures]:
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
            for (
                emb_feat_col_names
            ) in schema.embedding_feature_column_names.values():
                # _check_missing_columns() checks that vector columns are present,
                # hence I assume they are here
                col = emb_feat_col_names.vector_column_name
                if (
                    col in column_types
                    and column_types[col] not in allowed_vector_datatypes
                ):
                    wrong_type_vector_columns.append(col)

                if emb_feat_col_names.data_column_name:
                    col = emb_feat_col_names.data_column_name
                    if (
                        col in column_types
                        and column_types[col] not in allowed_data_datatypes
                    ):
                        wrong_type_data_columns.append(col)

                if emb_feat_col_names.link_to_data_column_name:
                    col = emb_feat_col_names.link_to_data_column_name
                    if (
                        col in column_types
                        and column_types[col]
                        not in allowed_link_to_data_datatypes
                    ):
                        wrong_type_link_to_data_columns.append(col)

            wrong_type_embedding_errors = []
            if wrong_type_vector_columns:
                wrong_type_embedding_errors.append(
                    InvalidTypeFeatures(
                        wrong_type_vector_columns,
                        expected_types=["list[float], np.array[float]"],
                    )
                )
            if wrong_type_data_columns:
                wrong_type_embedding_errors.append(
                    InvalidTypeFeatures(
                        wrong_type_data_columns, expected_types=["list[string]"]
                    )
                )
            if wrong_type_link_to_data_columns:
                wrong_type_embedding_errors.append(
                    InvalidTypeFeatures(
                        wrong_type_link_to_data_columns,
                        expected_types=["string"],
                    )
                )

            return (
                wrong_type_embedding_errors  # Will be empty list if no errors
            )

        return []

    @staticmethod
    def _check_type_tags(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeTags]:
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
                pa.null(),
            )
            wrong_type_cols = [
                col
                for col in _normalize_column_names(schema.tag_column_names)
                if col in column_types
                and column_types[col] not in allowed_datatypes
            ]
            if wrong_type_cols:
                return [
                    InvalidTypeTags(
                        wrong_type_cols, ["float", "int", "bool", "str"]
                    )
                ]
        return []

    @staticmethod
    def _check_type_shap_values(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeShapValues]:
        if schema.shap_values_column_names is not None:
            # should mirror server side
            allowed_datatypes = (
                pa.float64(),
                pa.int64(),
                pa.float32(),
                pa.int32(),
            )
            wrong_type_cols = [
                col
                for col in schema.shap_values_column_names.values()
                if col in column_types
                and column_types[col] not in allowed_datatypes
            ]
            if wrong_type_cols:
                return [
                    InvalidTypeShapValues(
                        wrong_type_cols, expected_types=["float", "int"]
                    )
                ]
        return []

    @staticmethod
    def _check_type_pred_act_labels(
        model_type: ModelTypes, schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
        errors = []
        columns = (
            ("Prediction labels", schema.prediction_label_column_name),
            ("Actual labels", schema.actual_label_column_name),
        )
        allowed_datatypes: tuple[Any, ...]
        if (
            model_type in CATEGORICAL_MODEL_TYPES
            or model_type == ModelTypes.GENERATIVE_LLM
        ):
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
                pa.null(),
            )
            for name, col in columns:
                if (
                    col is not None
                    and col in column_types
                    and column_types[col] not in allowed_datatypes
                ):
                    errors.append(
                        InvalidType(
                            name,
                            expected_types=["float", "int", "bool", "str"],
                            found_data_type=column_types[col],
                        )
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
                pa.null(),
            )
            for name, col in columns:
                if (
                    col is not None
                    and col in column_types
                    and column_types[col] not in allowed_datatypes
                ):
                    errors.append(
                        InvalidType(
                            name,
                            expected_types=["float", "int"],
                            found_data_type=column_types[col],
                        )
                    )
        return errors

    @staticmethod
    def _check_type_pred_act_scores(
        model_type: ModelTypes, schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
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
                pa.int8(),
                pa.int16(),
                pa.int32(),
                pa.int64(),
                pa.float32(),
                pa.float64(),
                pa.null(),
            )
            for name, col in columns:
                if (
                    col is not None
                    and col in column_types
                    and column_types[col] not in allowed_datatypes
                ):
                    errors.append(
                        InvalidType(
                            name,
                            expected_types=["float", "int"],
                            found_data_type=column_types[col],
                        )
                    )
        return errors

    @staticmethod
    def _check_type_multi_class_pred_threshold_act_scores(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
        """Check type for prediction / threshold / actual scores for multiclass model.

        Expect the scores to be a list of pyarrow structs that contains field
        "class_name" and field "score", where class_name is a string and score
        is a number.
        Example: '[{"class_name": "class1", "score": 0.1}, ...]'
        """
        errors = []
        columns = (
            ("Prediction scores", schema.prediction_score_column_name),
            (
                "Threshold scores",
                schema.multi_class_threshold_scores_column_name,
            ),
            ("Actual scores", schema.actual_score_column_name),
        )
        allowed_class_types = (pa.string(),)
        allowed_score_types = (
            pa.float64(),
            pa.float32(),
            pa.float16(),
            pa.int64(),
            pa.int32(),
            pa.int16(),
            pa.int8(),
        )
        # python dictionary is recognized as pyarrow struct
        allowed_class_score_map_datatypes = tuple(
            pa.list_(
                pa.struct([("class_name", class_type), ("score", score_type)])
            )
            for class_type in allowed_class_types
            for score_type in allowed_score_types
        )

        for name, col in columns:
            if (
                col is None
            ):  # multi_class_threshold_scores_column_name is optional
                continue
            if (
                col in column_types
                and column_types[col] not in allowed_class_score_map_datatypes
            ):
                errors.append(
                    InvalidType(
                        name,
                        expected_types=[
                            "List[Dict{class_name: str, score: int}]",
                            "List[Dict{class_name: str, score: float}]",
                        ],
                        found_data_type=column_types[col],
                    )
                )
        return errors

    @staticmethod
    def _check_type_prompt_response(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
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
        wrong_type_str_columns = []
        for field in fields_to_check:
            # If we pass a column name as string, said column must contain only strings as prompt/response
            if isinstance(field, str):
                if field in column_types and column_types[field] != pa.string():
                    wrong_type_str_columns.append(field)
            # If we pass a column names in a EmbeddingColumnNames object,
            # we validate the column type of both vector and data
            elif isinstance(field, EmbeddingColumnNames):
                # _check_missing_columns() checks that vector columns are present,
                # hence we assume they are here
                col = field.vector_column_name
                if (
                    col in column_types
                    and column_types[col] not in allowed_vector_datatypes
                ):
                    wrong_type_vector_columns.append(col)

                if field.data_column_name:
                    col = field.data_column_name
                    if (
                        col in column_types
                        and column_types[col] not in allowed_data_datatypes
                    ):
                        wrong_type_data_columns.append(col)

        wrong_type_col_errors = []
        if wrong_type_vector_columns:
            wrong_type_col_errors.append(
                InvalidTypeColumns(
                    wrong_type_vector_columns,
                    expected_types=["list[float], np.array[float]"],
                )
            )
        if wrong_type_data_columns:
            wrong_type_col_errors.append(
                InvalidTypeColumns(
                    wrong_type_data_columns, expected_types=["str, list[str]"]
                )
            )
        if wrong_type_str_columns:
            wrong_type_col_errors.append(
                InvalidTypeColumns(
                    wrong_type_str_columns, expected_types=["str"]
                )
            )

        return wrong_type_col_errors  # Will be empty list if no errors

    @staticmethod
    def _check_type_llm_prompt_templates(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
        if schema.prompt_template_column_names is None:
            return []

        allowed_datatypes = (pa.string(),)
        wrong_type_cols = []
        # Check type of template_column
        if schema.prompt_template_column_names.template_column_name is not None:
            col = schema.prompt_template_column_names.template_column_name
            if (
                col in column_types
                and column_types[col] not in allowed_datatypes
            ):
                wrong_type_cols.append(col)
        # Check type of template_column_version
        if (
            schema.prompt_template_column_names.template_version_column_name
            is not None
        ):
            col = (
                schema.prompt_template_column_names.template_version_column_name
            )
            if (
                col in column_types
                and column_types[col] not in allowed_datatypes
            ):
                wrong_type_cols.append(col)

        # Return errors if any
        if wrong_type_cols:
            return [
                InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols,
                    expected_types=["string"],
                )
            ]
        return []

    @staticmethod
    def _check_type_llm_config(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
        if schema.llm_config_column_names is None:
            return []

        allowed_datatypes = (pa.string(),)
        wrong_type_cols = []
        # Check type of model_column
        if schema.llm_config_column_names.model_column_name is not None:
            col = schema.llm_config_column_names.model_column_name
            if (
                col in column_types
                and column_types[col] not in allowed_datatypes
            ):
                wrong_type_cols.append(col)
        # Check type of params_column.
        # We are assuming at this point we have turned the dictionaries into json
        if schema.llm_config_column_names.params_column_name is not None:
            col = schema.llm_config_column_names.params_column_name
            if (
                col in column_types
                and column_types[col] not in allowed_datatypes
            ):
                wrong_type_cols.append(col)

        # Return errors if any
        if wrong_type_cols:
            return [
                InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols,
                    expected_types=["string"],
                )
            ]
        return []

    @staticmethod
    def _check_type_llm_run_metadata(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
        if schema.llm_run_metadata_column_names is None:
            return []

        allowed_datatypes = (
            pa.int8(),
            pa.int16(),
            pa.int32(),
            pa.int64(),
            pa.float32(),
            pa.float64(),
            pa.null(),
        )
        wrong_type_cols = []
        if schema.tag_column_names:
            tag_cols = _normalize_column_names(schema.tag_column_names)
            if (LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME in tag_cols) and (
                LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME in column_types
                and column_types[LLM_RUN_METADATA_TOTAL_TOKEN_COUNT_TAG_NAME]
                not in allowed_datatypes
            ):
                wrong_type_cols.append(
                    schema.llm_run_metadata_column_names.total_token_count_column_name
                )
            if (LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME in tag_cols) and (
                LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME in column_types
                and column_types[LLM_RUN_METADATA_PROMPT_TOKEN_COUNT_TAG_NAME]
                not in allowed_datatypes
            ):
                wrong_type_cols.append(
                    schema.llm_run_metadata_column_names.prompt_token_count_column_name
                )
            if (
                LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME in tag_cols
            ) and (
                LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME in column_types
                and column_types[LLM_RUN_METADATA_RESPONSE_TOKEN_COUNT_TAG_NAME]
                not in allowed_datatypes
            ):
                wrong_type_cols.append(
                    schema.llm_run_metadata_column_names.response_token_count_column_name
                )
            if (LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME in tag_cols) and (
                LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME in column_types
                and column_types[LLM_RUN_METADATA_RESPONSE_LATENCY_MS_TAG_NAME]
                not in allowed_datatypes
            ):
                wrong_type_cols.append(
                    schema.llm_run_metadata_column_names.response_latency_ms_column_name
                )

            # Return errors if there are any
            if wrong_type_cols:
                return [
                    InvalidTypeColumns(
                        wrong_type_columns=wrong_type_cols,  # type: ignore[arg-type]
                        expected_types=["int", "float"],
                    )
                ]
        return []

    @staticmethod
    def _check_type_retrieved_document_ids(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
        col = schema.retrieved_document_ids_column_name
        if col in column_types:
            # should mirror server side
            allowed_datatypes = (
                pa.list_(pa.string()),
                pa.null(),
            )
            if column_types[col] not in allowed_datatypes:
                return [
                    InvalidType(
                        "Retrieved Document IDs",
                        expected_types=["List[str]"],
                        found_data_type=column_types[col],
                    )
                ]
        return []

    @staticmethod
    def _check_type_image_segment_coordinates(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
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
            coord_col = schema.object_detection_prediction_column_names.bounding_boxes_coordinates_column_name
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

        if schema.object_detection_actual_column_names is not None:
            coord_col = schema.object_detection_actual_column_names.bounding_boxes_coordinates_column_name
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

        if schema.semantic_segmentation_prediction_column_names is not None:
            coord_col = schema.semantic_segmentation_prediction_column_names.polygon_coordinates_column_name
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

        if schema.semantic_segmentation_actual_column_names is not None:
            coord_col = schema.semantic_segmentation_actual_column_names.polygon_coordinates_column_name
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

        if schema.instance_segmentation_prediction_column_names is not None:
            inst_seg_pred = schema.instance_segmentation_prediction_column_names
            polygons_coord_col = inst_seg_pred.polygon_coordinates_column_name
            if (
                polygons_coord_col in column_types
                and column_types[polygons_coord_col]
                not in allowed_coordinate_types
            ):
                wrong_type_cols.append(polygons_coord_col)

            bbox_coord_col = (
                inst_seg_pred.bounding_boxes_coordinates_column_name
            )
            if (
                bbox_coord_col in column_types
                and column_types[bbox_coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(bbox_coord_col)

        if schema.instance_segmentation_actual_column_names is not None:
            inst_seg_actual = schema.instance_segmentation_actual_column_names
            coord_col = inst_seg_actual.polygon_coordinates_column_name
            if (
                coord_col in column_types
                and column_types[coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(coord_col)

            bbox_coord_col = (
                inst_seg_actual.bounding_boxes_coordinates_column_name
            )
            if (
                bbox_coord_col in column_types
                and column_types[bbox_coord_col] not in allowed_coordinate_types
            ):
                wrong_type_cols.append(bbox_coord_col)

        return (
            [
                InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols,
                    expected_types=["List[List[float]]"],
                )
            ]
            if wrong_type_cols
            else []
        )

    @staticmethod
    def _check_type_image_segment_categories(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
        # should mirror server side
        allowed_category_datatypes = (
            pa.list_(pa.string()),
        )  # List of categories
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

        if schema.semantic_segmentation_prediction_column_names is not None:
            cat_col_name = schema.semantic_segmentation_prediction_column_names.categories_column_name
            if (
                cat_col_name in column_types
                and column_types[cat_col_name] not in allowed_category_datatypes
            ):
                wrong_type_cols.append(cat_col_name)

        if schema.semantic_segmentation_actual_column_names is not None:
            cat_col_name = schema.semantic_segmentation_actual_column_names.categories_column_name
            if (
                cat_col_name in column_types
                and column_types[cat_col_name] not in allowed_category_datatypes
            ):
                wrong_type_cols.append(cat_col_name)

        if schema.instance_segmentation_prediction_column_names is not None:
            cat_col_name = schema.instance_segmentation_prediction_column_names.categories_column_name
            if (
                cat_col_name in column_types
                and column_types[cat_col_name] not in allowed_category_datatypes
            ):
                wrong_type_cols.append(cat_col_name)

        if schema.instance_segmentation_actual_column_names is not None:
            cat_col_name = schema.instance_segmentation_actual_column_names.categories_column_name
            if (
                cat_col_name in column_types
                and column_types[cat_col_name] not in allowed_category_datatypes
            ):
                wrong_type_cols.append(cat_col_name)

        return (
            [
                InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols,
                    expected_types=["List[str]"],
                )
            ]
            if wrong_type_cols
            else []
        )

    @staticmethod
    def _check_type_image_segment_scores(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
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
            score_col_name = (
                schema.object_detection_actual_column_names.scores_column_name
            )
            if (
                score_col_name is not None
                and score_col_name in column_types
                and column_types[score_col_name] not in allowed_score_datatypes
            ):
                wrong_type_cols.append(score_col_name)

        if schema.instance_segmentation_prediction_column_names is not None:
            score_col_name = schema.instance_segmentation_prediction_column_names.scores_column_name
            if (
                score_col_name is not None
                and score_col_name in column_types
                and column_types[score_col_name] not in allowed_score_datatypes
            ):
                wrong_type_cols.append(score_col_name)

        return (
            [
                InvalidTypeColumns(
                    wrong_type_columns=wrong_type_cols,
                    expected_types=["List[float]"],
                )
            ]
            if wrong_type_cols
            else []
        )

    # ------------
    # Value checks
    # ------------

    @staticmethod
    def _check_embedding_vectors_dimensionality(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[ValidationError]:
        if schema.embedding_feature_column_names is None:
            return []

        cols_to_check = []
        for (
            emb_feat_col_names
        ) in schema.embedding_feature_column_names.values():
            col = emb_feat_col_names.vector_column_name
            cols_to_check.append(col)

        (
            invalid_low_dim_vector_cols,
            invalid_high_dim_vector_cols,
        ) = _check_value_vector_dimensionality_helper(dataframe, cols_to_check)

        return (
            [
                InvalidValueEmbeddingVectorDimensionality(
                    invalid_low_dim_vector_cols,
                    invalid_high_dim_vector_cols,
                ),
            ]
            if invalid_low_dim_vector_cols or invalid_high_dim_vector_cols
            else []
        )

    @staticmethod
    def _check_embedding_raw_data_characters(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[ValidationError]:
        if schema.embedding_feature_column_names is None:
            return []

        cols_to_check = []
        for (
            emb_feat_col_names
        ) in schema.embedding_feature_column_names.values():
            col = emb_feat_col_names.data_column_name
            if col:
                cols_to_check.append(col)
        (
            invalid_long_string_data_cols,
            truncated_long_string_data_cols,
        ) = _check_value_raw_data_length_helper(dataframe, cols_to_check)

        if invalid_long_string_data_cols:
            return [
                InvalidValueEmbeddingRawDataTooLong(
                    invalid_long_string_data_cols
                )
            ]
        if truncated_long_string_data_cols:
            logger.warning(
                get_truncation_warning_message(
                    "Embedding raw data fields",
                    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
                )
            )
        return []

    @staticmethod
    def _check_value_rank(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidRankValue]:
        col = schema.rank_column_name
        lbound, ubound = (1, 100)

        if col is not None and col in dataframe.columns:
            rank_min_max = dataframe[col].agg(["min", "max"])
            if rank_min_max["min"] < lbound or rank_min_max["max"] > ubound:
                return [InvalidRankValue(col, "1-100")]
        return []

    @staticmethod
    def _check_id_field_str_length(
        dataframe: pd.DataFrame, schema_name: str, id_col_name: str | None
    ) -> list[ValidationError]:
        """Require prediction_id to be a string of length between MIN and MAX.

        Between MIN_PREDICTION_ID_LEN and MAX_PREDICTION_ID_LEN.
        """
        # We check whether the column name can be None is allowed in `Validator.validate_params`
        if id_col_name is None:
            return []

        # _check_value_missing will return error if there are missing values in the id fields
        # We can then proceed to check the character count of the values that are not None or missing.
        if id_col_name in dataframe.columns and not (
            # Check that the non-None values of the desired colum have a
            # string length between min_len and max_len
            # Does not check the None values
            dataframe[~dataframe[id_col_name].isnull()][id_col_name]
            .astype(str)
            .str.len()
            .between(MIN_PREDICTION_ID_LEN, MAX_PREDICTION_ID_LEN)
            .all()
        ):
            return [
                InvalidStringLengthInColumn(
                    schema_name=schema_name,
                    col_name=id_col_name,
                    min_length=MIN_PREDICTION_ID_LEN,
                    max_length=MAX_PREDICTION_ID_LEN,
                )
            ]
        return []

    @staticmethod
    def _check_document_id_field_str_length(
        dataframe: pd.DataFrame, schema_name: str, id_col_name: str | None
    ) -> list[ValidationError]:
        """Require document id to be a string of length between MIN and MAX.

        Between MIN_DOCUMENT_ID_LEN and MAX_DOCUMENT_ID_LEN.
        """
        # We check whether the column name can be None is allowed in `Validator.validate_params`
        if id_col_name is None:
            return []

        # _check_value_missing will return error if there are missing values in the id fields
        # We can then proceed to check the character count of the values that are not None or missing.
        if id_col_name in dataframe.columns and not (
            # Check that the non-None values of the desired colum have a
            # string length between min_len and max_len
            # Does not check the None values
            dataframe[~dataframe[id_col_name].isnull()][id_col_name]
            .astype(str)
            .str.len()
            .between(MIN_DOCUMENT_ID_LEN, MAX_DOCUMENT_ID_LEN)
            .all()
        ):
            return [
                InvalidStringLengthInColumn(
                    schema_name=schema_name,
                    col_name=id_col_name,
                    min_length=MIN_DOCUMENT_ID_LEN,
                    max_length=MAX_DOCUMENT_ID_LEN,
                )
            ]
        return []

    @staticmethod
    def _valid_char_limit(
        col_name: str, dataframe: pd.DataFrame, min_len: int, max_len: int
    ) -> bool:
        if not (
            col_name is not None
            and col_name in dataframe.columns
            and len(dataframe)
        ):
            return True
        return bool(
            dataframe[col_name]
            .astype(str)
            .str.len()
            .between(min_len, max_len)
            .all()
        )

    @staticmethod
    def _check_value_tag(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidTagLength]:
        if schema.tag_column_names is None:
            return []

        wrong_tag_cols = []
        truncated_tag_cols = []
        for col in _normalize_column_names(schema.tag_column_names):
            # This is to be defensive, validate_params should guarantee that this column is in
            # the dataframe, via _check_missing_columns, and return an error before reaching this
            # block if not
            # Checks max tag length when any values in a column are strings
            if (
                col in dataframe.columns
                and dataframe[col].map(type).eq(str).any()  # type: ignore[arg-type]
            ):
                max_tag_len = (
                    dataframe[col]
                    .apply(_check_value_string_length_helper)
                    .max()
                )
                if max_tag_len > MAX_TAG_LENGTH:
                    wrong_tag_cols.append(col)
                elif max_tag_len > MAX_TAG_LENGTH_TRUNCATION:
                    truncated_tag_cols.append(col)
        if wrong_tag_cols:
            return [InvalidTagLength(wrong_tag_cols)]
        if truncated_tag_cols:
            logger.warning(
                get_truncation_warning_message(
                    "tags", MAX_TAG_LENGTH_TRUNCATION
                )
            )
        return []

    @staticmethod
    def _check_value_ranking_category(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidValueMissingValue | InvalidRankingCategoryValue]:
        col: str | None
        if schema.relevance_labels_column_name is not None:
            col = schema.relevance_labels_column_name
        elif schema.attributions_column_name is not None:
            col = schema.attributions_column_name
        else:
            col = schema.actual_label_column_name
        if col is not None and col in dataframe.columns:
            if dataframe[col].isnull().any():
                # do not attach duplicated missing value error
                # which would be caught by _check_value_missing
                return []
            if dataframe[col].astype(str).str.len().min() == 0:
                return [InvalidRankingCategoryValue(col)]
            # empty list
            not_null_filter = dataframe[col].notnull()
            if dataframe[not_null_filter][col].map(len).min() == 0:
                return [InvalidValueMissingValue(col, "empty list")]
            # no empty string in list
            if (
                dataframe[not_null_filter][col]
                .map(lambda x: (min(map(len, x))))
                .min()
                == 0
            ):
                return [InvalidRankingCategoryValue(col)]
        return []

    @staticmethod
    def _check_length_multi_class_maps(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidNumClassesMultiClassMap]:
        # each entry in column is a list of dictionaries mapping class names and scores
        # validate length of list of dictionaries for each column
        invalid_cols = {}
        cols = [
            schema.prediction_score_column_name,
            schema.multi_class_threshold_scores_column_name,
            schema.actual_score_column_name,
        ]
        for col in cols:
            if (
                col is None
            ):  # multi_class_threshold_scores_column_name is optional
                continue
            invalid_num_classes = []
            for list_of_dicts_class_names_to_scores in dataframe[col]:
                if (
                    list_of_dicts_class_names_to_scores is None
                    or len(list_of_dicts_class_names_to_scores) == 0
                ):
                    invalid_num_classes.append("0")
                if (
                    len(list_of_dicts_class_names_to_scores)
                    > MAX_NUMBER_OF_MULTI_CLASS_CLASSES
                ):
                    invalid_num_classes.append(
                        str(len(list_of_dicts_class_names_to_scores))
                    )
            if invalid_num_classes:
                invalid_cols[col] = invalid_num_classes
        if invalid_cols:
            return [InvalidNumClassesMultiClassMap(invalid_cols)]
        return []

    @staticmethod
    def _check_classes_and_scores_values_in_multi_class_maps(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[
        InvalidMultiClassClassNameLength
        | InvalidMultiClassActScoreValue
        | InvalidMultiClassPredScoreValue
    ]:
        """Validate the class names and score values of dictionaries.

        - class name length
        - valid actual score
        - valid prediction / threshold score
        """
        cols = [
            schema.prediction_score_column_name,
            schema.multi_class_threshold_scores_column_name,
            schema.actual_score_column_name,
        ]
        invalid_class_names = {}
        invalid_pred_scores = {}
        lbound, ubound = (0, 1)
        invalid_actual_scores = False
        errors: list[
            InvalidMultiClassClassNameLength
            | InvalidMultiClassActScoreValue
            | InvalidMultiClassPredScoreValue
        ] = []
        for col in cols:
            if (
                col is None
            ):  # multi_class_threshold_scores_column_name is optional
                continue
            invalid_class_names_for_col = set()
            # example list_class_name_and_scores_dicts:
            # List[Dict{"class_name": "1", "score": 0.1}, Dict{"class_name": "2", "score": 0.2} ...]
            for list_class_name_and_scores_dicts in dataframe[col]:
                # json_normalize explodes dictionaries to dataframe where columns are class_name and score
                class_names_and_scores_df = pd.json_normalize(
                    list_class_name_and_scores_dicts
                )
                # get list of class_names and scores by extracting column in df
                class_names = class_names_and_scores_df["class_name"]
                scores = class_names_and_scores_df["score"]
                # validate class name lengths
                for class_name in class_names:
                    if (
                        len(class_name) == 0
                        or len(class_name) > MAX_MULTI_CLASS_NAME_LENGTH
                    ):
                        invalid_class_names_for_col.add(class_name)
                if invalid_class_names_for_col:
                    invalid_class_names[col] = invalid_class_names_for_col
                # validate class scores
                if (
                    col == schema.actual_score_column_name
                ):  # actual scores must be 0 or 1
                    if any(
                        score != lbound and score != ubound for score in scores
                    ):
                        invalid_actual_scores = True
                else:  # pred / thresh scores must be between 0 and 1
                    invalid_scores_for_col = set()
                    score_min_max = scores.agg(["min", "max"])
                    if score_min_max["min"] < lbound:
                        invalid_scores_for_col.add(str(score_min_max["min"]))
                    if score_min_max["max"] > ubound:
                        invalid_scores_for_col.add(str(score_min_max["max"]))
                    if any(score is None for score in scores):
                        invalid_scores_for_col.add("None")
                    if any(math.isnan(score) for score in scores):
                        invalid_scores_for_col.add("nan")
                    if invalid_scores_for_col:
                        invalid_pred_scores[col] = invalid_scores_for_col
        if invalid_class_names:
            errors.append(InvalidMultiClassClassNameLength(invalid_class_names))
        if invalid_pred_scores:
            errors.append(InvalidMultiClassPredScoreValue(invalid_pred_scores))  # type: ignore[arg-type]
        if invalid_actual_scores:
            errors.append(InvalidMultiClassActScoreValue(col))  # type: ignore[arg-type, arg-type]
        return errors

    @staticmethod
    def _check_each_multi_class_pred_has_threshold(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidMultiClassThresholdClasses]:
        """Validate threshold scores for Multi Class models.

        If threshold scores column is included in schema and dataframe, validate that
        for each prediction score received, the associated threshold score for that
        class was also received.
        """
        threshold_col = schema.multi_class_threshold_scores_column_name
        if threshold_col is None:
            return []
        for index, class_thresh_score_set in enumerate(
            dataframe[threshold_col]
        ):
            threshold_scores = pd.json_normalize(class_thresh_score_set)
            threshold_classes = threshold_scores["class_name"]
            thresh_class_set = set(threshold_classes)
            class_pred_score_set = dataframe[
                schema.prediction_score_column_name
            ][index]
            pred_scores = pd.json_normalize(class_pred_score_set)
            pred_classes = pred_scores["class_name"]
            pred_class_set = set(pred_classes)
            if pred_class_set != thresh_class_set:
                return [
                    InvalidMultiClassThresholdClasses(
                        threshold_col, pred_class_set, thresh_class_set
                    )
                ]
        return []

    @staticmethod
    def _check_value_timestamp(
        dataframe: pd.DataFrame,
        schema: Schema,
    ) -> list[InvalidValueMissingValue | InvalidValueTimestamp]:
        # Due to the timing difference between checking this here and the data finally
        # hitting the same check on server side, there's a some chance for a false
        # result, i.e. the check here succeeds but the same check on server side fails.
        col = schema.timestamp_column_name
        if col is not None and col in dataframe.columns:
            # When a timestamp column has Date and NaN, pyarrow will be fine, but
            # pandas min/max will fail due to type incompatibility. So we check for
            # missing value first.
            if dataframe[col].isnull().any():
                return [
                    InvalidValueMissingValue("Prediction timestamp", "missing")
                ]

            now_t = datetime.now(tz=timezone.utc)
            lbound, ubound = (
                (
                    now_t
                    - timedelta(days=MAX_PAST_YEARS_FROM_CURRENT_TIME * 365)
                ).timestamp(),
                (
                    now_t
                    + timedelta(days=MAX_FUTURE_YEARS_FROM_CURRENT_TIME * 365)
                ).timestamp(),
            )
            # faster than pyarrow compute
            stats = dataframe[col].agg(["min", "max"])

            ta = pa.Table.from_pandas(stats.to_frame())
            type_ = ta.column(0).type
            min_, max_ = ta.column(0)
            # Add warning when future timestamps are sent
            if (
                (
                    isinstance(type_, pa.TimestampType)
                    and stats["max"].timestamp() > now_t.timestamp()
                )
                or (
                    type_ in (pa.int64(), pa.float64())
                    and max_.as_py() > now_t.timestamp()
                )
                or (
                    type_ == pa.date32()
                    and (
                        int(max_.cast(pa.int32()).as_py() * 60 * 60 * 24)
                        > now_t.timestamp()
                    )
                )
                or (
                    type_ == pa.date64()
                    and (
                        int(max_.cast(pa.int64()).as_py() // 1000)
                        > now_t.timestamp()
                    )
                )
            ):
                logger.warning(
                    "Caution when sending predictions with future timestamps."
                    "Arize only stores 2 years worth of data. For example, if you sent predictions "
                    "to Arize from 1.5 years ago, and now send predictions with timestamps of a year in "
                    "the future, the oldest 0.5 years will be dropped to maintain the 2 years worth of data "
                    "requirement."
                )

            # this part needs improvement: dealing with python types is hard :(
            # Return error if timestamp is out of range
            if (
                (
                    isinstance(type_, pa.TimestampType)
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
                        int(min_.cast(pa.int32()).as_py() * 60 * 60 * 24)
                        < lbound
                        or int(max_.cast(pa.int32()).as_py() * 60 * 60 * 24)
                        > ubound
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
                return [InvalidValueTimestamp(timestamp_col_name=col)]

        return []

    # _check_invalid_missing_values validates that columns that cannot have any null values
    # do not have any null values and returns an error if they do
    @staticmethod
    def _check_invalid_missing_values(
        dataframe: pd.DataFrame, schema: BaseSchema, model_type: ModelTypes
    ) -> list[InvalidValueMissingValue]:
        errors = []
        columns: tuple[tuple[str, str | None], ...] = ()
        if isinstance(schema, CorpusSchema):
            columns = (("Document ID", schema.document_id_column_name),)
        elif isinstance(schema, Schema):
            if model_type == ModelTypes.RANKING:
                columns = (
                    ("Prediction ID", schema.prediction_id_column_name),
                    (
                        "Prediction Group ID",
                        schema.prediction_group_id_column_name,
                    ),
                    ("Rank", schema.rank_column_name),
                )
            else:
                columns = (("Prediction ID", schema.prediction_id_column_name),)
        for name, col in columns:
            if col is not None and col in dataframe.columns:
                if dataframe[col].isnull().any():
                    errors.append(
                        InvalidValueMissingValue(
                            name, wrong_values="missing", column=col
                        )
                    )
                elif (
                    dataframe[col].dtype
                    in (np.dtype("float64"), np.dtype("float32"))
                    and np.isinf(dataframe[col]).any()
                ):
                    errors.append(
                        InvalidValueMissingValue(
                            name, wrong_values="infinite", column=col
                        )
                    )
        return errors

    # _check_invalid_record_prod validates there's not a single row in the dataframe
    # with pred_label, pred_score, actual_label, actual_score, and shap_value
    # columns all evaluates to null and returns an error with the row numbers
    # where that is the case
    @staticmethod
    def _check_invalid_record_prod(
        dataframe: pd.DataFrame,
        environment: Environments,
        schema: Schema,
        model_type: ModelTypes,
    ) -> list[InvalidRecord]:
        if environment in (Environments.VALIDATION, Environments.TRAINING):
            return []

        if (
            model_type in CATEGORICAL_MODEL_TYPES
            or model_type in NUMERIC_MODEL_TYPES
        ):
            columns_to_validate = [
                schema.prediction_label_column_name,
                schema.prediction_score_column_name,
                schema.actual_label_column_name,
                schema.actual_score_column_name,
            ]
        elif model_type == ModelTypes.RANKING:
            columns_to_validate = [
                schema.prediction_label_column_name,
                schema.prediction_score_column_name,
                schema.prediction_group_id_column_name,
                schema.rank_column_name,
                schema.actual_label_column_name,
                schema.actual_score_column_name,
                schema.relevance_score_column_name,
                schema.relevance_labels_column_name,
            ]
        else:
            columns_to_validate = []
        # TODO: add separate logic for objective detection and generative model types

        if schema.shap_values_column_names is not None:
            columns_to_validate.extend(
                list(schema.shap_values_column_names.values())
            )

        return Validator._check_invalid_record_helper(
            dataframe, columns_to_validate
        )

    @staticmethod
    def _check_invalid_record_preprod(
        dataframe: pd.DataFrame,
        environment: Environments,
        schema: Schema,
        model_type: ModelTypes,
    ) -> list[InvalidRecord]:
        """Validates there's not a single row in the dataframe with all nulls.

        Returns errors if any row has all of pred_label and pred_score evaluating to
        null, OR all of actual_label and actual_score evaluating to null.
        """
        if environment == Environments.PRODUCTION:
            return []

        if (
            model_type in CATEGORICAL_MODEL_TYPES
            or model_type in NUMERIC_MODEL_TYPES
        ):
            pred_columns_to_validate = [
                schema.prediction_label_column_name,
                schema.prediction_score_column_name,
            ]
            actual_columns_to_validate = [
                schema.actual_label_column_name,
                schema.actual_score_column_name,
            ]
        elif model_type == ModelTypes.RANKING:
            pred_columns_to_validate = [
                schema.prediction_label_column_name,
                schema.prediction_score_column_name,
                schema.prediction_group_id_column_name,
                schema.rank_column_name,
            ]
            actual_columns_to_validate = [
                schema.actual_label_column_name,
                schema.actual_score_column_name,
                schema.relevance_score_column_name,
                schema.relevance_labels_column_name,
            ]
        else:
            pred_columns_to_validate = []
            actual_columns_to_validate = []
            # TODO: add separate logic for objective detection and generative model types

        return Validator._check_invalid_record_helper(
            dataframe, pred_columns_to_validate
        ) + Validator._check_invalid_record_helper(
            dataframe, actual_columns_to_validate
        )

    @staticmethod
    def _check_invalid_record_helper(
        dataframe: pd.DataFrame, column_names: list[str | None]
    ) -> list[InvalidRecord]:
        """Check that there are no null values in a subset of columns.

        The column subset is computed from the input list of columns `column_names`
        that are not None and that are present in the dataframe. Returns an error if
        null values are found.

        Returns:
            List[InvalidRecord]: An error expressing the rows that are problematic

        """
        columns_subset = [
            col
            for col in column_names
            if col is not None and col in dataframe.columns
        ]
        if len(columns_subset) == 0:
            return []
        null_filter = dataframe[columns_subset].isnull().all(axis=1)
        null_index = null_filter[null_filter].index.values
        if len(null_index) == 0:
            return []
        return [InvalidRecord(columns_subset, null_index)]  # type: ignore[arg-type]

    @staticmethod
    def _check_type_prediction_group_id(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
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
                    InvalidType(
                        "prediction_group_ids",
                        expected_types=["str", "int"],
                        found_data_type=column_types[col],
                    )
                ]
        return []

    @staticmethod
    def _check_type_rank(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
        col = schema.rank_column_name
        if col in column_types:
            allowed_datatypes = (
                pa.int64(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
            )
            if column_types[col] not in allowed_datatypes:
                return [
                    InvalidType(
                        "rank",
                        expected_types=["int"],
                        found_data_type=column_types[col],
                    )
                ]
        return []

    @staticmethod
    def _check_type_ranking_category(
        schema: Schema, column_types: dict[str, Any]
    ) -> list[InvalidType]:
        col: str | None
        if schema.relevance_labels_column_name is not None:
            col = schema.relevance_labels_column_name
        elif schema.attributions_column_name is not None:
            col = schema.attributions_column_name
        else:
            col = schema.actual_label_column_name
        if col is not None and col in column_types:
            allowed_datatypes = (pa.list_(pa.string()), pa.string(), pa.null())
            if column_types[col] not in allowed_datatypes:
                return [
                    InvalidType(
                        "relevance labels column for ranking models",
                        expected_types=["list of string", "string"],
                        found_data_type=column_types[col],
                    )
                ]
        return []

    @staticmethod
    def _check_value_bounding_boxes_coordinates(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidBoundingBoxesCoordinates]:
        errors = []
        if schema.object_detection_prediction_column_names is not None:
            coords_col_name = schema.object_detection_prediction_column_names.bounding_boxes_coordinates_column_name  # noqa: E501
            error = _check_value_bounding_boxes_coordinates_helper(
                dataframe[coords_col_name]
            )
            if error is not None:
                errors.append(error)
        if schema.object_detection_actual_column_names is not None:
            coords_col_name = schema.object_detection_actual_column_names.bounding_boxes_coordinates_column_name  # noqa: E501
            error = _check_value_bounding_boxes_coordinates_helper(
                dataframe[coords_col_name]
            )
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_bounding_boxes_categories(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidBoundingBoxesCategories]:
        errors = []
        if schema.object_detection_prediction_column_names is not None:
            cat_col_name = schema.object_detection_prediction_column_names.categories_column_name
            error = _check_value_bounding_boxes_categories_helper(
                dataframe[cat_col_name]
            )
            if error is not None:
                errors.append(error)
        if schema.object_detection_actual_column_names is not None:
            cat_col_name = schema.object_detection_actual_column_names.categories_column_name
            error = _check_value_bounding_boxes_categories_helper(
                dataframe[cat_col_name]
            )
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_bounding_boxes_scores(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidBoundingBoxesScores]:
        errors = []
        if schema.object_detection_prediction_column_names is not None:
            sc_col_name = schema.object_detection_prediction_column_names.scores_column_name
            if sc_col_name is not None:
                error = _check_value_bounding_boxes_scores_helper(
                    dataframe[sc_col_name]
                )
                if error is not None:
                    errors.append(error)
        if schema.object_detection_actual_column_names is not None:
            sc_col_name = (
                schema.object_detection_actual_column_names.scores_column_name
            )
            if sc_col_name is not None:
                error = _check_value_bounding_boxes_scores_helper(
                    dataframe[sc_col_name]
                )
                if error is not None:
                    errors.append(error)
        return errors

    @staticmethod
    def _check_value_semantic_segmentation_polygon_coordinates(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidPolygonCoordinates]:
        errors = []
        if schema.semantic_segmentation_prediction_column_names is not None:
            coords_col_name = schema.semantic_segmentation_prediction_column_names.polygon_coordinates_column_name  # noqa: E501
            error = _check_value_polygon_coordinates_helper(
                dataframe[coords_col_name]
            )
            if error is not None:
                errors.append(error)
        if schema.semantic_segmentation_actual_column_names is not None:
            coords_col_name = schema.semantic_segmentation_actual_column_names.polygon_coordinates_column_name
            error = _check_value_polygon_coordinates_helper(
                dataframe[coords_col_name]
            )
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_semantic_segmentation_polygon_categories(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidPolygonCategories]:
        errors = []
        if schema.semantic_segmentation_prediction_column_names is not None:
            cat_col_name = schema.semantic_segmentation_prediction_column_names.categories_column_name
            error = _check_value_polygon_categories_helper(
                dataframe[cat_col_name]
            )
            if error is not None:
                errors.append(error)
        if schema.semantic_segmentation_actual_column_names is not None:
            cat_col_name = schema.semantic_segmentation_actual_column_names.categories_column_name
            error = _check_value_polygon_categories_helper(
                dataframe[cat_col_name]
            )
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_instance_segmentation_polygon_coordinates(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidPolygonCoordinates]:
        errors = []
        if schema.instance_segmentation_prediction_column_names is not None:
            coords_col_name = schema.instance_segmentation_prediction_column_names.polygon_coordinates_column_name  # noqa: E501
            error = _check_value_polygon_coordinates_helper(
                dataframe[coords_col_name]
            )
            if error is not None:
                errors.append(error)
        if schema.instance_segmentation_actual_column_names is not None:
            coords_col_name = schema.instance_segmentation_actual_column_names.polygon_coordinates_column_name
            error = _check_value_polygon_coordinates_helper(
                dataframe[coords_col_name]
            )
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_instance_segmentation_polygon_categories(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidPolygonCategories]:
        errors = []
        if schema.instance_segmentation_prediction_column_names is not None:
            cat_col_name = schema.instance_segmentation_prediction_column_names.categories_column_name
            error = _check_value_polygon_categories_helper(
                dataframe[cat_col_name]
            )
            if error is not None:
                errors.append(error)
        if schema.instance_segmentation_actual_column_names is not None:
            cat_col_name = schema.instance_segmentation_actual_column_names.categories_column_name
            error = _check_value_polygon_categories_helper(
                dataframe[cat_col_name]
            )
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _check_value_instance_segmentation_polygon_scores(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidPolygonScores]:
        errors = []
        if schema.instance_segmentation_prediction_column_names is not None:
            sc_col_name = schema.instance_segmentation_prediction_column_names.scores_column_name
            if sc_col_name is not None:
                error = _check_value_polygon_scores_helper(
                    dataframe[sc_col_name]
                )
                if error is not None:
                    errors.append(error)
        return errors

    @staticmethod
    def _check_value_instance_segmentation_bbox_coordinates(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidBoundingBoxesCoordinates]:
        errors = []
        if schema.instance_segmentation_prediction_column_names is not None:
            coords_col_name = schema.instance_segmentation_prediction_column_names.bounding_boxes_coordinates_column_name  # noqa: E501
            if coords_col_name is not None:
                error = _check_value_bounding_boxes_coordinates_helper(
                    dataframe[coords_col_name]
                )
                if error is not None:
                    errors.append(error)
        if schema.instance_segmentation_actual_column_names is not None:
            coords_col_name = schema.instance_segmentation_actual_column_names.bounding_boxes_coordinates_column_name  # noqa: E501
            if coords_col_name is not None:
                error = _check_value_bounding_boxes_coordinates_helper(
                    dataframe[coords_col_name]
                )
                if error is not None:
                    errors.append(error)
        return errors

    @staticmethod
    def _check_value_prompt_response(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[ValidationError]:
        vector_cols_to_check = []
        text_cols_to_check = []
        if isinstance(schema.prompt_column_names, str):
            text_cols_to_check.append(schema.prompt_column_names)
        elif isinstance(schema.prompt_column_names, EmbeddingColumnNames):
            vector_cols_to_check.append(
                schema.prompt_column_names.vector_column_name
            )
            if schema.prompt_column_names.data_column_name is not None:
                text_cols_to_check.append(
                    schema.prompt_column_names.data_column_name
                )

        if isinstance(schema.response_column_names, str):
            text_cols_to_check.append(schema.response_column_names)
        elif isinstance(schema.response_column_names, EmbeddingColumnNames):
            vector_cols_to_check.append(
                schema.response_column_names.vector_column_name
            )
            if schema.response_column_names.data_column_name is not None:
                text_cols_to_check.append(
                    schema.response_column_names.data_column_name
                )

        (
            invalid_long_string_data_cols,
            truncated_long_string_data_cols,
        ) = _check_value_raw_data_length_helper(dataframe, text_cols_to_check)
        (
            invalid_low_dim_vector_cols,
            invalid_high_dim_vector_cols,
        ) = _check_value_vector_dimensionality_helper(
            dataframe, vector_cols_to_check
        )

        errors: list[ValidationError] = []
        if invalid_long_string_data_cols:
            errors.append(
                InvalidValueEmbeddingRawDataTooLong(
                    invalid_long_string_data_cols
                )
            )
        if invalid_low_dim_vector_cols or invalid_high_dim_vector_cols:
            errors.append(
                InvalidValueEmbeddingVectorDimensionality(  # type: ignore[arg-type]
                    invalid_low_dim_vector_cols,
                    invalid_high_dim_vector_cols,
                )
            )
        if errors:
            return errors
        if truncated_long_string_data_cols:
            logger.warning(
                get_truncation_warning_message(
                    "prompt and response text fields",
                    MAX_RAW_DATA_CHARACTERS_TRUNCATION,
                )
            )

        return []

    @staticmethod
    def _check_value_llm_model_name(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidStringLengthInColumn]:
        if schema.llm_config_column_names is None:
            return []
        col = schema.llm_config_column_names.model_column_name
        if col is not None:
            max_len = (
                dataframe[col].apply(_check_value_string_length_helper).max()
            )
            if max_len > MAX_LLM_MODEL_NAME_LENGTH:
                return [
                    InvalidStringLengthInColumn(
                        schema_name="llm_config_column_names.model_column_name",
                        col_name=col,
                        min_length=0,
                        max_length=MAX_LLM_MODEL_NAME_LENGTH,
                    )
                ]
            if max_len > MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION:
                logger.warning(
                    get_truncation_warning_message(
                        "LLM model names", MAX_LLM_MODEL_NAME_LENGTH_TRUNCATION
                    )
                )
        return []

    @staticmethod
    def _check_value_llm_prompt_template(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidStringLengthInColumn]:
        if schema.prompt_template_column_names is None:
            return []
        col = schema.prompt_template_column_names.template_column_name
        if col is not None:
            max_len = (
                dataframe[col].apply(_check_value_string_length_helper).max()
            )
            if max_len > MAX_PROMPT_TEMPLATE_LENGTH:
                return [
                    InvalidStringLengthInColumn(
                        schema_name="prompt_template_column_names.template_column_name",
                        col_name=col,
                        min_length=0,
                        max_length=MAX_PROMPT_TEMPLATE_LENGTH,
                    )
                ]
            if max_len > MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION:
                logger.warning(
                    get_truncation_warning_message(
                        "prompt templates",
                        MAX_PROMPT_TEMPLATE_LENGTH_TRUNCATION,
                    )
                )
        return []

    @staticmethod
    def _check_value_llm_prompt_template_version(
        dataframe: pd.DataFrame, schema: Schema
    ) -> list[InvalidStringLengthInColumn]:
        if schema.prompt_template_column_names is None:
            return []
        col = schema.prompt_template_column_names.template_version_column_name
        if col is not None:
            max_len = (
                dataframe[col].apply(_check_value_string_length_helper).max()
            )
            if max_len > MAX_PROMPT_TEMPLATE_VERSION_LENGTH:
                return [
                    InvalidStringLengthInColumn(
                        schema_name="prompt_template_column_names.template_version_column_name",
                        col_name=col,
                        min_length=0,
                        max_length=MAX_PROMPT_TEMPLATE_VERSION_LENGTH,
                    )
                ]
            if max_len > MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION:
                logger.warning(
                    get_truncation_warning_message(
                        "prompt template versions",
                        MAX_PROMPT_TEMPLATE_VERSION_LENGTH_TRUNCATION,
                    )
                )
        return []

    @staticmethod
    def _check_type_document_columns(
        schema: CorpusSchema, column_types: dict[str, Any]
    ) -> list[InvalidTypeColumns]:
        invalid_types = []
        allowed_datatypes: tuple[Any, ...]
        # Check document id
        col = schema.document_id_column_name
        if col in column_types:
            allowed_datatypes = (
                pa.string(),
                pa.int64(),
                pa.int32(),
                pa.int16(),
                pa.int8(),
            )
            if column_types[col] not in allowed_datatypes:
                invalid_types += [
                    InvalidTypeColumns(
                        wrong_type_columns=[col],
                        expected_types=["str", "int"],
                    )
                ]

        # Check document version
        col = schema.document_version_column_name
        if col in column_types:
            allowed_datatype = pa.string()
            if column_types[col] != allowed_datatype:
                invalid_types += [
                    InvalidTypeColumns(
                        wrong_type_columns=[col],
                        expected_types=["str"],
                    )
                ]

        if not schema.document_text_embedding_column_names:
            return invalid_types

        # Check document embedding vector
        col = schema.document_text_embedding_column_names.vector_column_name
        if col in column_types:
            allowed_datatypes = (
                pa.list_(pa.float64()),
                pa.list_(pa.float32()),
            )
            if column_types[col] not in allowed_datatypes:
                invalid_types += [
                    InvalidTypeColumns(
                        wrong_type_columns=[col],
                        expected_types=["list[float], np.array[float]"],
                    )
                ]

        # Check document embedding data
        col = schema.document_text_embedding_column_names.data_column_name
        if col in column_types:
            allowed_datatypes = (
                pa.string(),  # Text
                pa.list_(pa.string()),  # Token Array
            )
            if column_types[col] not in allowed_datatypes:
                invalid_types += [
                    InvalidTypeColumns(
                        wrong_type_columns=[col],
                        expected_types=["list[str]"],
                    )
                ]

        # Check document embedding link to data
        col = (
            schema.document_text_embedding_column_names.link_to_data_column_name
        )
        if col in column_types:
            allowed_datatypes = (pa.string(),)
            if column_types[col] not in allowed_datatypes:
                invalid_types += [
                    InvalidTypeColumns(
                        wrong_type_columns=[col],
                        expected_types=["str"],
                    )
                ]

        if invalid_types:
            return invalid_types

        return []


def _check_value_string_length_helper(x: object) -> int:
    if isinstance(x, str):
        return len(x)
    return 0


def _check_value_vector_dimensionality_helper(
    dataframe: pd.DataFrame, cols_to_check: list[str]
) -> tuple[list[str], list[str]]:
    invalid_low_dimensionality_vector_cols = []
    invalid_high_dimensionality_vector_cols = []
    for col in cols_to_check:
        vector_dims = dataframe[col].apply(
            lambda vec: 0 if vec is None or vec is np.nan else len(vec)
        )
        if (vector_dims == 1).any():
            invalid_low_dimensionality_vector_cols.append(col)
        if (vector_dims > MAX_EMBEDDING_DIMENSIONALITY).any():
            invalid_high_dimensionality_vector_cols.append(col)

    return (
        invalid_low_dimensionality_vector_cols,
        invalid_high_dimensionality_vector_cols,
    )


def _check_value_raw_data_length_helper(
    dataframe: pd.DataFrame, cols_to_check: list[str]
) -> tuple[list[str], list[str]]:
    invalid_long_string_data_cols = []
    truncated_long_string_data_cols = []
    for col in cols_to_check:
        try:
            max_data_len = (
                dataframe[col]
                .apply(
                    lambda data: 0
                    if data is None
                    else _count_characters_raw_data(data)
                )
                .max()
            )
        except TypeError as exc:
            e = TypeError(f"Cannot validate the column '{col}'. " + str(exc))
            logger.exception(e)
            raise e from exc
        if max_data_len > MAX_RAW_DATA_CHARACTERS:
            invalid_long_string_data_cols.append(col)
        elif max_data_len > MAX_RAW_DATA_CHARACTERS_TRUNCATION:
            truncated_long_string_data_cols.append(col)
    return invalid_long_string_data_cols, truncated_long_string_data_cols


def _check_value_bounding_boxes_coordinates_helper(
    coordinates_col: pd.Series,
) -> InvalidBoundingBoxesCoordinates | None:
    def check(boxes: object) -> None:
        # We allow for zero boxes. None coordinates list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if boxes is None:
            raise InvalidBoundingBoxesCoordinates(reason="none_boxes")
        # Type ignore: boxes comes from pandas Series, validated at runtime to be iterable
        for box in boxes:  # type: ignore[attr-defined]
            if box is None or len(box) == 0:
                raise InvalidBoundingBoxesCoordinates(
                    reason="none_or_empty_box"
                )
            error = _box_coordinates_wrong_format(box)
            if error is not None:
                raise error

    try:
        coordinates_col.apply(check)
    except InvalidBoundingBoxesCoordinates as e:
        return e
    return None


def _box_coordinates_wrong_format(
    box_coords: object,
) -> InvalidBoundingBoxesCoordinates | None:
    if (
        # Coordinates should be a collection of 4 floats
        len(box_coords) != 4  # type: ignore[arg-type]
        # Coordinates should be positive
        # Type ignore: box_coords validated at runtime to be iterable/indexable
        or any(k < 0 for k in box_coords)  # type: ignore[attr-defined]
        # Coordinates represent the top-left & bottom-right corners of a box: x1 < x2
        or box_coords[0] >= box_coords[2]  # type: ignore[index]
        # Coordinates represent the top-left & bottom-right corners of a box: y1 < y2
        or box_coords[1] >= box_coords[3]  # type: ignore[index]
    ):
        return InvalidBoundingBoxesCoordinates(
            reason="boxes_coordinates_wrong_format"
        )
    return None


def _check_value_bounding_boxes_categories_helper(
    categories_col: pd.Series,
) -> InvalidBoundingBoxesCategories | None:
    def check(categories: object) -> None:
        # We allow for zero boxes. None category list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if categories is None:
            raise InvalidBoundingBoxesCategories(reason="none_category_list")
        # Type ignore: categories validated at runtime to be iterable
        for category in categories:  # type: ignore[attr-defined]
            # Allow for empty string category, no None values
            if category is None:
                raise InvalidBoundingBoxesCategories(reason="none_category")

    try:
        categories_col.apply(check)
    except InvalidBoundingBoxesCategories as e:
        return e
    return None


def _check_value_bounding_boxes_scores_helper(
    scores_col: pd.Series,
) -> InvalidBoundingBoxesScores | None:
    def check(scores: object) -> None:
        # We allow for zero boxes. None confidence score list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if scores is None:
            raise InvalidBoundingBoxesScores(reason="none_score_list")
        # Type ignore: scores validated at runtime to be iterable
        for score in scores:  # type: ignore[attr-defined]
            # Confidence scores are between 0 and 1
            if score < 0 or score > 1:
                raise InvalidBoundingBoxesScores(reason="scores_out_of_bounds")

    try:
        scores_col.apply(check)
    except InvalidBoundingBoxesScores as e:
        return e
    return None


def _polygon_coordinates_wrong_format(
    polygon_coords: object,
) -> InvalidPolygonCoordinates | None:
    """Check if polygon coordinates are valid.

    Validates:
    - Has at least 3 vertices (6 coordinates)
    - All coordinates are positive
    - Number of coordinates is even (pairs of x,y)
    - No repeated vertices
    - No self-intersections in the polygon

    Args:
        polygon_coords: List of coordinates [x1, y1, x2, y2, ...]

    Returns:
        Error object if invalid, None if valid
    """
    # Basic validations
    if (
        # Coordinates should be a collection of more than 6 floats (3 pairs of x,y coordinates)
        len(polygon_coords) < 6  # type: ignore[arg-type]
        # Coordinates should be positive
        # Type ignore: polygon_coords validated at runtime to be iterable
        or any(k < 0 for k in polygon_coords)  # type: ignore[arg-type, attr-defined]
        # Coordinates should be a collection of pairs of floats
        or len(polygon_coords) % 2 != 0  # type: ignore[arg-type]
    ):
        return InvalidPolygonCoordinates(
            reason="polygon_coordinates_wrong_format",
            coordinates=polygon_coords,  # type: ignore[arg-type]
        )

    # Convert flat list to list of points [(x1,y1), (x2,y2), ...]
    coords_seq = cast("Sequence[float]", polygon_coords)
    points = [
        (coords_seq[i], coords_seq[i + 1]) for i in range(0, len(coords_seq), 2)
    ]

    # Check for repeated vertices
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] == points[j]:
                return InvalidPolygonCoordinates(
                    reason="polygon_coordinates_repeated_vertices",
                    coordinates=polygon_coords,  # type: ignore[arg-type]
                )

    # Check for self-intersections
    # We need to check if any two non-adjacent edges intersect
    edges = [
        (points[i], points[(i + 1) % len(points)]) for i in range(len(points))
    ]

    for i in range(len(edges)):
        for j in range(i + 2, len(edges)):
            # Skip adjacent edges
            if i == 0 and j == len(edges) - 1:
                continue

            # Check if edges intersect
            if segments_intersect(
                edges[i][0], edges[i][1], edges[j][0], edges[j][1]
            ):
                return InvalidPolygonCoordinates(
                    reason="polygon_coordinates_self_intersecting_vertices",
                    coordinates=polygon_coords,  # type: ignore[arg-type]
                )

    return None


def _check_value_polygon_coordinates_helper(
    coordinates_col: pd.Series,
) -> InvalidPolygonCoordinates | None:
    def check(polygons: object) -> None:
        # We allow for zero polygons. None coordinates list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if polygons is None:
            raise InvalidPolygonCoordinates(reason="none_polygons")
        # Type ignore: polygons validated at runtime to be iterable
        for polygon in polygons:  # type: ignore[attr-defined]
            if polygon is None or len(polygon) == 0:
                raise InvalidPolygonCoordinates(reason="none_or_empty_polygon")
            error = _polygon_coordinates_wrong_format(polygon)
            if error is not None:
                raise error

    try:
        coordinates_col.apply(check)
    except InvalidPolygonCoordinates as e:
        return e
    return None


def _check_value_polygon_categories_helper(
    categories_col: pd.Series,
) -> InvalidPolygonCategories | None:
    def check(categories: object) -> None:
        # We allow for zero boxes. None category list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if categories is None:
            raise InvalidPolygonCategories(reason="none_category_list")
        # Type ignore: categories validated at runtime to be iterable
        for category in categories:  # type: ignore[attr-defined]
            # Allow for empty string category, no None values
            if category is None:
                raise InvalidPolygonCategories(reason="none_category")

    try:
        categories_col.apply(check)
    except InvalidPolygonCategories as e:
        return e
    return None


def _check_value_polygon_scores_helper(
    scores_col: pd.Series,
) -> InvalidPolygonScores | None:
    def check(scores: object) -> None:
        # We allow for zero boxes. None confidence score list is not allowed (will break following tests:
        # 'NoneType is not iterable')
        if scores is None:
            raise InvalidPolygonScores(reason="none_score_list")
        # Type ignore: scores validated at runtime to be iterable
        for score in scores:  # type: ignore[attr-defined]
            # Confidence scores are between 0 and 1
            if score < 0 or score > 1:
                raise InvalidPolygonScores(reason="scores_out_of_bounds")

    try:
        scores_col.apply(check)
    except InvalidPolygonScores as e:
        return e
    return None


def _count_characters_raw_data(data: str | list[str]) -> int:
    character_count = 0
    if isinstance(data, str):
        character_count = len(data)
    elif is_iterable_of(data, str):
        for string in data:
            character_count += len(string)
    else:
        raise TypeError(
            f"Cannot count characters for raw data. Expecting strings or "
            f"list of strings but another type was found: {type(data)}."
        )
    return character_count
