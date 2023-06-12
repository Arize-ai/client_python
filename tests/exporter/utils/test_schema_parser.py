import pandas as pd
from arize.exporter.utils.schema_parser import get_arize_schema
from arize.pandas.logger import Schema
from arize.utils.types import ObjectDetectionColumnNames


def test_classification_model():
    classification_df = pd.DataFrame(
        [
            {
                "this_is_a_tag__tag": 49,
                "predictionID": "pred_id_1",
                "categoricalActualLabel": "fraud",
                "categoricalPredictionLabel": "not_fraud",
                "scorePredictionLabel": 0.9,
                "scoreActualLabel": 1.0,
                "time": "2023-04-10 00:00:00+0000",
                "feature_1": "123",
                "feature_2": 123,
            }
        ]
    )

    schema = get_arize_schema(classification_df)

    assert isinstance(schema, Schema)

    assert schema.asdict() == {
        "actual_label_column_name": "categoricalActualLabel",
        "actual_score_column_name": "scoreActualLabel",
        "attributions_column_name": None,
        "embedding_feature_column_names": {},
        "feature_column_names": None,
        "object_detection_actual_column_names": None,
        "object_detection_prediction_column_names": None,
        "prediction_group_id_column_name": None,
        "prediction_id_column_name": "predictionID",
        "prediction_label_column_name": "categoricalPredictionLabel",
        "prediction_score_column_name": "scorePredictionLabel",
        "prompt_column_names": None,
        "rank_column_name": None,
        "relevance_labels_column_name": None,
        "relevance_score_column_name": None,
        "response_column_names": None,
        "shap_values_column_names": None,
        "tag_column_names": ["this_is_a_tag__tag"],
        "timestamp_column_name": "time",
    }


def test_object_detection():
    od_df = pd.DataFrame(
        [
            {
                "boxPredictionCoordinates": [
                    [608.81146, 265.98914, 624.9569, 355.17545],
                    [1, 2, 3, 4],
                ],
                "boxActualLabels": ["banana", "grapes"],
                "boxPredictionLabels": ["apples"],
                "boxPredictionScores": [0.5],
                "time": "2023-04-10 00:00:00+0000",
                "predictionID": "002b4c71-d335-47f0-92d3-4fd2b337ac95",
                "boxActualCoordinates": [
                    [579.06226, 46.043736, 592.70807, 106.160675],
                    [75.80126, 90.106964, 113.96147, 396.76505],
                ],
                "feature_1": "asdf",
            }
        ]
    )
    schema = get_arize_schema(od_df)

    assert isinstance(schema, Schema)
    schema.object_detection_actual_column_names

    assert schema.asdict() == {
        "actual_label_column_name": None,
        "actual_score_column_name": None,
        "attributions_column_name": None,
        "embedding_feature_column_names": {},
        "feature_column_names": None,
        "object_detection_actual_column_names": ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="boxActualCoordinates",
            categories_column_name="boxActualLabels",
            scores_column_name=None,
        ),
        "object_detection_prediction_column_names": ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="boxPredictionCoordinates",
            categories_column_name=None,
            scores_column_name="boxPredictionScores",
        ),
        "prediction_group_id_column_name": None,
        "prediction_id_column_name": "predictionID",
        "prediction_label_column_name": None,
        "prediction_score_column_name": None,
        "prompt_column_names": None,
        "rank_column_name": None,
        "relevance_labels_column_name": None,
        "relevance_score_column_name": None,
        "response_column_names": None,
        "shap_values_column_names": None,
        "tag_column_names": [],
        "timestamp_column_name": "time",
    }


def test_embeddings():
    embedding_df = pd.DataFrame(
        [
            {
                "time": "2023-04-10 00:00:00+0000",
                "embedding1__embVector": [1, 2, 3],
                "embedding1__rawData": "Test sentence",
                "embedding2__embVector": [4, 5, 6],
                "embedding2__linkToData": "www.google.com",
                "predictionID": "002b4c71-d335-47f0-92d3-4fd2b337ac95",
            }
        ]
    )
    schema = get_arize_schema(embedding_df)

    assert isinstance(schema, Schema)
    schema.object_detection_actual_column_names

    assert schema.asdict() == {
        "actual_label_column_name": None,
        "actual_score_column_name": None,
        "attributions_column_name": None,
        "embedding_feature_column_names": {
            "embedding1": {
                "vector_column_name": "embedding1__embVector",
                "data_column_name": "embedding1__rawData",
                "link_to_data_column_name": None,
            },
            "embedding2": {
                "vector_column_name": "embedding2__embVector",
                "data_column_name": None,
                "link_to_data_column_name": "embedding2__linkToData",
            },
        },
        "feature_column_names": None,
        "object_detection_actual_column_names": None,
        "object_detection_prediction_column_names": None,
        "prediction_group_id_column_name": None,
        "prediction_id_column_name": "predictionID",
        "prediction_label_column_name": None,
        "prediction_score_column_name": None,
        "prompt_column_names": None,
        "rank_column_name": None,
        "relevance_labels_column_name": None,
        "relevance_score_column_name": None,
        "response_column_names": None,
        "shap_values_column_names": None,
        "tag_column_names": [],
        "timestamp_column_name": "time",
    }


def test_rankings():
    embedding_df = pd.DataFrame(
        [
            {
                "predictionID": "0",
                "time": "2023-04-11 18:00:00+0000",
                "predictionGroupID": "0",
                "ranking:rank": 1,
                "rank": 2,  # not used for rank
                "modelVersion": "no_version",
                "ranking:relevance": 0.0,
                "ranking:label": "asdf",
            }
        ]
    )
    schema = get_arize_schema(embedding_df)

    assert isinstance(schema, Schema)
    schema.object_detection_actual_column_names

    assert schema.asdict() == {
        "actual_label_column_name": None,
        "actual_score_column_name": None,
        "attributions_column_name": None,
        "embedding_feature_column_names": {},
        "feature_column_names": None,
        "object_detection_actual_column_names": None,
        "object_detection_prediction_column_names": None,
        "prediction_group_id_column_name": "predictionGroupID",
        "prediction_id_column_name": "predictionID",
        "prediction_label_column_name": None,
        "prediction_score_column_name": None,
        "prompt_column_names": None,
        "rank_column_name": "ranking:rank",
        "relevance_labels_column_name": "ranking:label",
        "relevance_score_column_name": "ranking:relevance",
        "response_column_names": None,
        "shap_values_column_names": None,
        "tag_column_names": [],
        "timestamp_column_name": "time",
    }


def test_llm():
    embedding_df = pd.DataFrame(
        [
            {
                "predictionID": "0",
                "time": "2023-04-11 18:00:00+0000",
                "prompt": "blah",
                "response": "blahblah",
            }
        ]
    )
    schema = get_arize_schema(embedding_df)

    assert isinstance(schema, Schema)
    schema.object_detection_actual_column_names

    assert schema.asdict() == {
        "actual_label_column_name": None,
        "actual_score_column_name": None,
        "attributions_column_name": None,
        "embedding_feature_column_names": {},
        "feature_column_names": None,
        "object_detection_actual_column_names": None,
        "object_detection_prediction_column_names": None,
        "prediction_group_id_column_name": None,
        "prediction_id_column_name": "predictionID",
        "prediction_label_column_name": None,
        "prediction_score_column_name": None,
        "prompt_column_names": "prompt",
        "rank_column_name": None,
        "relevance_labels_column_name": None,
        "relevance_score_column_name": None,
        "response_column_names": "response",
        "shap_values_column_names": None,
        "tag_column_names": [],
        "timestamp_column_name": "time",
    }


def test_regression():
    embedding_df = pd.DataFrame(
        [
            {
                "predictionID": "0",
                "time": "2023-04-11 18:00:00+0000",
                "prompt": "blah",
                "response": "blahblah",
                "numericPredictionLabel": 123,
                "numericActualLabel": 456,
            }
        ]
    )
    schema = get_arize_schema(embedding_df)

    assert isinstance(schema, Schema)
    schema.object_detection_actual_column_names

    assert schema.asdict() == {
        "actual_label_column_name": None,
        "actual_score_column_name": "numericActualLabel",
        "attributions_column_name": None,
        "embedding_feature_column_names": {},
        "feature_column_names": None,
        "object_detection_actual_column_names": None,
        "object_detection_prediction_column_names": None,
        "prediction_group_id_column_name": None,
        "prediction_id_column_name": "predictionID",
        "prediction_label_column_name": None,
        "prediction_score_column_name": "numericPredictionLabel",
        "prompt_column_names": "prompt",
        "rank_column_name": None,
        "relevance_labels_column_name": None,
        "relevance_score_column_name": None,
        "response_column_names": "response",
        "shap_values_column_names": None,
        "tag_column_names": [],
        "timestamp_column_name": "time",
    }
