# type: ignore[pb2]
from typing import Optional

from .. import public_pb2 as pb2
from .types import (
    CATEGORICAL_MODEL_TYPES,
    NUMERIC_MODEL_TYPES,
    CorpusSchema,
    EmbeddingColumnNames,
    Environments,
    ModelTypes,
    Schema,
)


def _get_pb_schema(
    schema: Schema,
    model_id: str,
    model_version: Optional[str],
    model_type: ModelTypes,
    environment: Environments,
    batch_id: str,
):
    s = pb2.Schema()
    s.constants.model_id = model_id

    if model_version is not None:
        s.constants.model_version = model_version

    if environment == Environments.PRODUCTION:
        s.constants.environment = pb2.Schema.Environment.PRODUCTION
    elif environment == Environments.VALIDATION:
        s.constants.environment = pb2.Schema.Environment.VALIDATION
    elif environment == Environments.TRAINING:
        s.constants.environment = pb2.Schema.Environment.TRAINING
    else:
        raise ValueError(f"unexpected environment: {environment}")

    # Map user-friendly external model types -> internal model types when sending to Arize
    if model_type in NUMERIC_MODEL_TYPES:
        s.constants.model_type = pb2.Schema.ModelType.NUMERIC
    elif model_type in CATEGORICAL_MODEL_TYPES:
        s.constants.model_type = pb2.Schema.ModelType.SCORE_CATEGORICAL
    elif model_type == ModelTypes.RANKING:
        s.constants.model_type = pb2.Schema.ModelType.RANKING
    elif model_type == ModelTypes.OBJECT_DETECTION:
        s.constants.model_type = pb2.Schema.ModelType.OBJECT_DETECTION
    elif model_type == ModelTypes.GENERATIVE_LLM:
        s.constants.model_type = pb2.Schema.ModelType.GENERATIVE_LLM
    elif model_type == ModelTypes.MULTI_CLASS:
        s.constants.model_type = pb2.Schema.ModelType.MULTI_CLASS

    if batch_id is not None:
        s.constants.batch_id = batch_id

    if schema.prediction_id_column_name is not None:
        s.arrow_schema.prediction_id_column_name = (
            schema.prediction_id_column_name
        )

    if schema.timestamp_column_name is not None:
        s.arrow_schema.timestamp_column_name = schema.timestamp_column_name

    if schema.prediction_label_column_name is not None:
        s.arrow_schema.prediction_label_column_name = (
            schema.prediction_label_column_name
        )

    if (
        model_type == ModelTypes.OBJECT_DETECTION
        and schema.object_detection_prediction_column_names is not None
    ):
        s.arrow_schema.prediction_object_detection_label_column_names.bboxes_coordinates_column_name = schema.object_detection_prediction_column_names.bounding_boxes_coordinates_column_name  # noqa: E501
        s.arrow_schema.prediction_object_detection_label_column_names.bboxes_categories_column_name = schema.object_detection_prediction_column_names.categories_column_name  # noqa: E501
        if (
            schema.object_detection_prediction_column_names.scores_column_name
            is not None
        ):
            s.arrow_schema.prediction_object_detection_label_column_names.bboxes_scores_column_name = schema.object_detection_prediction_column_names.scores_column_name  # noqa: E501

    if schema.prediction_score_column_name is not None:
        if model_type in NUMERIC_MODEL_TYPES:
            # allow numeric prediction to be sent in as either prediction_label (legacy) or
            # prediction_score.
            s.arrow_schema.prediction_label_column_name = (
                schema.prediction_score_column_name
            )
        else:
            s.arrow_schema.prediction_score_column_name = (
                schema.prediction_score_column_name
            )

    if schema.feature_column_names is not None:
        s.arrow_schema.feature_column_names.extend(schema.feature_column_names)

    if schema.embedding_feature_column_names is not None:
        for (
            emb_name,
            emb_col_names,
        ) in schema.embedding_feature_column_names.items():
            # emb_name is how it will show in the UI
            s.arrow_schema.embedding_feature_column_names_map[
                emb_name
            ].vector_column_name = emb_col_names.vector_column_name
            if emb_col_names.data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    emb_name
                ].data_column_name = emb_col_names.data_column_name
            if emb_col_names.link_to_data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    emb_name
                ].link_to_data_column_name = (
                    emb_col_names.link_to_data_column_name
                )

    if schema.prompt_column_names is not None:
        if isinstance(schema.prompt_column_names, str):
            s.arrow_schema.embedding_feature_column_names_map[
                "prompt"
            ].data_column_name = schema.prompt_column_names
        elif isinstance(schema.prompt_column_names, EmbeddingColumnNames):
            col_names = schema.prompt_column_names
            s.arrow_schema.embedding_feature_column_names_map[
                "prompt"
            ].vector_column_name = col_names.vector_column_name
            if col_names.data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    "prompt"
                ].data_column_name = col_names.data_column_name
    if schema.response_column_names is not None:
        if isinstance(schema.response_column_names, str):
            s.arrow_schema.embedding_feature_column_names_map[
                "response"
            ].data_column_name = schema.response_column_names
        elif isinstance(schema.response_column_names, EmbeddingColumnNames):
            col_names = schema.response_column_names
            s.arrow_schema.embedding_feature_column_names_map[
                "response"
            ].vector_column_name = col_names.vector_column_name
            if col_names.data_column_name:
                s.arrow_schema.embedding_feature_column_names_map[
                    "response"
                ].data_column_name = col_names.data_column_name

    if schema.tag_column_names is not None:
        s.arrow_schema.tag_column_names.extend(schema.tag_column_names)

    if (
        model_type == ModelTypes.RANKING
        and schema.relevance_labels_column_name is not None
    ):
        s.arrow_schema.actual_label_column_name = (
            schema.relevance_labels_column_name
        )
    elif (
        model_type == ModelTypes.RANKING
        and schema.attributions_column_name is not None
    ):
        s.arrow_schema.actual_label_column_name = (
            schema.attributions_column_name
        )
    elif schema.actual_label_column_name is not None:
        s.arrow_schema.actual_label_column_name = (
            schema.actual_label_column_name
        )

    if (
        model_type == ModelTypes.RANKING
        and schema.relevance_score_column_name is not None
    ):
        s.arrow_schema.actual_score_column_name = (
            schema.relevance_score_column_name
        )
    elif schema.actual_score_column_name is not None:
        if model_type in NUMERIC_MODEL_TYPES:
            # allow numeric prediction to be sent in as either prediction_label (legacy) or
            # prediction_score.
            s.arrow_schema.actual_label_column_name = (
                schema.actual_score_column_name
            )
        else:
            s.arrow_schema.actual_score_column_name = (
                schema.actual_score_column_name
            )

    if schema.shap_values_column_names is not None:
        s.arrow_schema.shap_values_column_names.update(
            schema.shap_values_column_names
        )

    if schema.prediction_group_id_column_name is not None:
        s.arrow_schema.prediction_group_id_column_name = (
            schema.prediction_group_id_column_name
        )

    if schema.rank_column_name is not None:
        s.arrow_schema.rank_column_name = schema.rank_column_name

    if (
        model_type == ModelTypes.OBJECT_DETECTION
        and schema.object_detection_actual_column_names is not None
    ):
        s.arrow_schema.actual_object_detection_label_column_names.bboxes_coordinates_column_name = schema.object_detection_actual_column_names.bounding_boxes_coordinates_column_name  # noqa: E501
        s.arrow_schema.actual_object_detection_label_column_names.bboxes_categories_column_name = schema.object_detection_actual_column_names.categories_column_name  # noqa: E501
        if (
            schema.object_detection_actual_column_names.scores_column_name
            is not None
        ):
            s.arrow_schema.actual_object_detection_label_column_names.bboxes_scores_column_name = schema.object_detection_actual_column_names.scores_column_name  # noqa: E501

    if model_type == ModelTypes.GENERATIVE_LLM:
        if schema.prompt_template_column_names is not None:
            s.arrow_schema.prompt_template_column_names.template_column_name = (
                schema.prompt_template_column_names.template_column_name
            )
            s.arrow_schema.prompt_template_column_names.template_version_column_name = schema.prompt_template_column_names.template_version_column_name  # noqa: E501
        if schema.llm_config_column_names is not None:
            s.arrow_schema.llm_config_column_names.model_column_name = (
                schema.llm_config_column_names.model_column_name
            )
            s.arrow_schema.llm_config_column_names.params_map_column_name = (
                schema.llm_config_column_names.params_column_name
            )
        if schema.retrieved_document_ids_column_name is not None:
            s.arrow_schema.retrieved_document_ids_column_name = (
                schema.retrieved_document_ids_column_name
            )
    if model_type == ModelTypes.MULTI_CLASS:
        if schema.prediction_score_column_name is not None:
            s.arrow_schema.prediction_score_column_name = (
                schema.prediction_score_column_name
            )
        if schema.multi_class_threshold_scores_column_name is not None:
            s.arrow_schema.multi_class_threshold_scores_column_name = (
                schema.multi_class_threshold_scores_column_name
            )
        if schema.actual_score_column_name is not None:
            s.arrow_schema.actual_score_column_name = (
                schema.actual_score_column_name
            )
    return s


def _get_pb_schema_corpus(
    schema: CorpusSchema,
    model_id: str,
    model_type: ModelTypes,
    environment: Environments,
) -> pb2.Schema:
    s = pb2.Schema()
    s.constants.model_id = model_id
    if environment == Environments.CORPUS:
        s.constants.environment = pb2.Schema.Environment.CORPUS
    else:
        raise ValueError(f"unexpected environment: {environment}")
    if model_type == ModelTypes.GENERATIVE_LLM:
        s.constants.model_type = pb2.Schema.ModelType.GENERATIVE_LLM
    else:
        raise ValueError(
            f"unexpected model type for corpus environment: {model_type}"
        )

    if schema.document_id_column_name is not None:
        s.arrow_schema.document_column_names.id_column_name = (
            schema.document_id_column_name
        )
    if schema.document_version_column_name is not None:
        s.arrow_schema.document_column_names.version_column_name = (
            schema.document_version_column_name
        )
    if schema.document_text_embedding_column_names is not None:
        s.arrow_schema.document_column_names.text_column_name.vector_column_name = schema.document_text_embedding_column_names.vector_column_name  # noqa: E501
        s.arrow_schema.document_column_names.text_column_name.data_column_name = schema.document_text_embedding_column_names.data_column_name  # noqa: E501
        if (
            schema.document_text_embedding_column_names.link_to_data_column_name
            is not None
        ):
            s.arrow_schema.document_column_names.text_column_name.link_to_data_column_name = schema.document_text_embedding_column_names.link_to_data_column_name  # noqa: E501
    return s


def _get_pb_schema_tracing(
    model_id: str,
    model_version: Optional[str],
) -> pb2.Schema:
    s = pb2.Schema()
    s.constants.model_id = model_id
    s.constants.environment = pb2.Schema.Environment.TRACING
    s.constants.model_type = pb2.Schema.ModelType.GENERATIVE_LLM
    if model_version is not None:
        s.constants.model_version = model_version
    s.arize_spans.SetInParent()
    return s
