from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from arize.pandas.logger import Schema
from arize.utils.types import (
    EmbeddingColumnNames,
    InstanceSegmentationActualColumnNames,
    InstanceSegmentationPredictionColumnNames,
    ObjectDetectionColumnNames,
    SemanticSegmentationColumnNames,
)

PREDICTION_ID_COL = "predictionID"
TIME_COL = "time"


def _get_suffix(suffix):
    return lambda colname: colname[-1 * len(suffix) :] == suffix


def _get_prefix(prefix):
    return lambda colname: colname[: len(prefix)] == prefix


def _get_colname(colname) -> Callable:
    return lambda columns: colname if colname in columns else None


# Helper functions for data types:
is_vector = _get_suffix("__embVector")
is_raw_data = _get_suffix("__rawData")
is_link = _get_suffix("__linkToData")
is_tag = _get_suffix("__tag")
is_prompt = _get_prefix("prompt")
is_response = _get_prefix("response")


# Object detection columns:
get_box_pred_scores = _get_colname("boxPredictionScores")
get_box_pred_labels = _get_colname("boxPredictionLabel")

get_box_actual_scores = _get_colname("boxActualScores")
get_box_actual_labels = _get_colname("boxActualLabels")

# Semantic segmentation columns:
get_semantic_segmentation_polygon_pred_labels = _get_colname(
    "semanticSegmentationPolygonPredictionLabels"
)
get_semantic_segmentation_polygon_actual_labels = _get_colname(
    "semanticSegmentationPolygonActualLabels"
)

# Instance segmentation columns:
get_instance_segmentation_polygon_pred_scores = _get_colname(
    "instanceSegmentationPolygonPredictionScores"
)
get_instance_segmentation_polygon_pred_labels = _get_colname(
    "instanceSegmentationPolygonPredictionLabels"
)
get_instance_segmentation_polygon_pred_bbox_coordinates = _get_colname(
    "instanceSegmentationBoxPredictionCoordinates"
)

get_instance_segmentation_polygon_actual_labels = _get_colname(
    "instanceSegmentationPolygonActualLabels"
)
get_instance_segmentation_polygon_actual_bbox_coordinates = _get_colname(
    "instanceSegmentationBoxActualCoordinates"
)


# Classification columns:
get_pred_label = _get_colname("categoricalPredictionLabel")
get_pred_score = _get_colname("scorePredictionLabel")

get_actual_label = _get_colname("categoricalActualLabel")
get_actual_score = _get_colname("scoreActualLabel")


# Regression columns:
get_numeric_pred = _get_colname("numericPredictionLabel")
get_numeric_actual = _get_colname("numericActualLabel")


# Ranking columns:
get_pred_group_id = _get_colname("predictionGroupID")
get_ranking_category = _get_colname("ranking:category")
get_ranking_relevance = _get_colname("ranking:relevance")
get_ranking_label = _get_colname("ranking:label")
get_rank = _get_colname("ranking:rank")

# TODO: Multi Class columns


def get_tags(df_columns: Iterable) -> List[str]:
    return [c for c in df_columns if is_tag(c)]


def get_embedding_dict(
    vector_col_name: str, df_columns: Iterable[str]
) -> Dict[str, EmbeddingColumnNames]:
    embedding_dict = {"vector_column_name": vector_col_name}

    prefix = vector_col_name.split("__embVector")[0]
    for other_col in df_columns:
        if prefix in other_col:
            if is_raw_data(other_col):
                embedding_dict["data_column_name"] = other_col
            elif is_link(other_col):
                embedding_dict["link_to_data_column_name"] = other_col

    return {prefix: EmbeddingColumnNames(**embedding_dict)}


def get_embeddings(
    df_columns: Iterable[str],
) -> Dict[str, EmbeddingColumnNames]:
    embed_dict = {}
    for col in df_columns:
        # We exclude prompt/response from embedding features
        if is_vector(col) and not (is_prompt(col) or is_response(col)):
            single_embed = get_embedding_dict(col, df_columns)
            embed_dict = {**embed_dict, **single_embed}

    return embed_dict


def _get_prompt_or_response(
    prefix: str, df_columns: Iterable[str]
) -> Optional[EmbeddingColumnNames]:
    vec = _get_colname(f"{prefix}__embVector")(df_columns)
    if vec is None:
        return None

    embedding_dict = {"vector_column_name": vec}
    if f"{prefix}__rawData" in df_columns:
        embedding_dict["data_column_name"] = _get_colname(f"{prefix}__rawData")(
            df_columns
        )
    if f"{prefix}__linkToData" in df_columns:
        embedding_dict["link_to_data_column_name"] = _get_colname(
            f"{prefix}__linkToData"
        )(df_columns)

    return EmbeddingColumnNames(**embedding_dict)


def get_prompt(df_columns: Iterable[str]) -> Optional[EmbeddingColumnNames]:
    return _get_prompt_or_response("prompt", df_columns)


def get_response(df_columns: Iterable[str]) -> Optional[EmbeddingColumnNames]:
    return _get_prompt_or_response("response", df_columns)


def get_object_detection_prediction(
    cols: Iterable[str],
) -> Optional[ObjectDetectionColumnNames]:
    if "boxPredictionCoordinates" in cols:
        return ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="boxPredictionCoordinates",
            categories_column_name=get_box_pred_labels(cols),
            scores_column_name=get_box_pred_scores(cols),
        )


def get_object_detection_actual(
    cols: Iterable[str],
) -> Optional[ObjectDetectionColumnNames]:
    if "boxActualCoordinates" in cols:
        return ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="boxActualCoordinates",
            categories_column_name=get_box_actual_labels(cols),
            scores_column_name=get_box_actual_scores(cols),
        )


def get_semantic_segmentation_prediction(
    cols: Iterable[str],
) -> Optional[SemanticSegmentationColumnNames]:
    if "semanticSegmentationPolygonPredictionCoordinates" in cols:
        return SemanticSegmentationColumnNames(
            polygon_coordinates_column_name="semanticSegmentationPolygonPredictionCoordinates",
            categories_column_name=get_semantic_segmentation_polygon_pred_labels(
                cols
            ),
        )


def get_semantic_segmentation_actual(
    cols: Iterable[str],
) -> Optional[SemanticSegmentationColumnNames]:
    if "semanticSegmentationPolygonActualCoordinates" in cols:
        return SemanticSegmentationColumnNames(
            polygon_coordinates_column_name="semanticSegmentationPolygonActualCoordinates",
            categories_column_name=get_semantic_segmentation_polygon_actual_labels(
                cols
            ),
        )


def get_instance_segmentation_prediction(
    cols: Iterable[str],
) -> Optional[InstanceSegmentationPredictionColumnNames]:
    if "instanceSegmentationPolygonPredictionCoordinates" in cols:
        return InstanceSegmentationPredictionColumnNames(
            polygon_coordinates_column_name="instanceSegmentationPolygonPredictionCoordinates",
            categories_column_name=get_instance_segmentation_polygon_pred_labels(
                cols
            ),
            scores_column_name=get_instance_segmentation_polygon_pred_scores(
                cols
            ),
            bounding_boxes_coordinates_column_name=get_instance_segmentation_polygon_pred_bbox_coordinates(
                cols
            ),
        )


def get_instance_segmentation_actual(
    cols: Iterable[str],
) -> Optional[InstanceSegmentationActualColumnNames]:
    if "instanceSegmentationPolygonActualCoordinates" in cols:
        return InstanceSegmentationActualColumnNames(
            polygon_coordinates_column_name="instanceSegmentationPolygonActualCoordinates",
            categories_column_name=get_instance_segmentation_polygon_actual_labels(
                cols
            ),
            bounding_boxes_coordinates_column_name=get_instance_segmentation_polygon_actual_bbox_coordinates(
                cols
            ),
        )


def get_arize_schema(df: pd.DataFrame) -> Schema:
    """
    Take a dataframe returned from the flight exporter and turn it into an arize schema
    """
    cols = df.columns

    prediction_score_column_name = get_pred_score(cols) or get_numeric_pred(
        cols
    )
    actual_score_column_name = get_actual_score(cols) or get_numeric_actual(
        cols
    )

    schema = Schema(
        prediction_id_column_name=PREDICTION_ID_COL,
        tag_column_names=get_tags(cols),
        timestamp_column_name=TIME_COL,
        prediction_label_column_name=get_pred_label(cols),
        prediction_score_column_name=prediction_score_column_name,
        actual_label_column_name=get_actual_label(cols),
        actual_score_column_name=actual_score_column_name,
        embedding_feature_column_names=get_embeddings(cols),
        object_detection_actual_column_names=get_object_detection_actual(cols),
        object_detection_prediction_column_names=get_object_detection_prediction(
            cols
        ),
        semantic_segmentation_actual_column_names=get_semantic_segmentation_actual(
            cols
        ),
        semantic_segmentation_prediction_column_names=get_semantic_segmentation_prediction(
            cols
        ),
        instance_segmentation_actual_column_names=get_instance_segmentation_actual(
            cols
        ),
        instance_segmentation_prediction_column_names=get_instance_segmentation_prediction(
            cols
        ),
        prediction_group_id_column_name=get_pred_group_id(cols),
        rank_column_name=get_rank(cols),
        relevance_score_column_name=get_ranking_relevance(cols),
        relevance_labels_column_name=get_ranking_label(cols),
        prompt_column_names=get_prompt(cols),
        response_column_names=get_response(cols),
    )
    return schema
