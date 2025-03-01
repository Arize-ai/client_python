# type: ignore[pb2]
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

import pandas as pd

from arize.pandas.logger import Client
from arize.utils.types import (
    Environments,
    ModelTypes,
    Schema,
)


class IntegrationClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        space_id: Optional[str] = None,
        space_key: Optional[str] = None,
        uri: Optional[str] = "https://api.arize.com/v1",
        additional_headers: Optional[Dict[str, str]] = None,
        request_verify: Union[bool, str] = True,
        developer_key: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """
        Wrapper for the Arize Client specific to WhyLabs profiles.
        """
        self._client = Client(
            api_key=api_key,
            space_id=space_id,
            space_key=space_key,
            uri=uri,
            additional_headers=additional_headers,
            request_verify=request_verify,
            developer_key=developer_key,
            host=host,
            port=port,
        )

    def create_model(
        self,
        model_id: str,
        model_type: ModelTypes,
        environment: Environments = Environments.PRODUCTION,
    ) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now() - timedelta(days=365)],
                "ARIZE_PLACEHOLDER_STRING": "ARIZE_DUMMY",
                "ARIZE_PLACEHOLDER_FLOAT": 1,
            }
        )

        if model_type == ModelTypes.RANKING:
            schema = Schema(
                timestamp_column_name="timestamp",
                prediction_group_id_column_name="ARIZE_PLACEHOLDER_FLOAT",
                rank_column_name="ARIZE_PLACEHOLDER_FLOAT",
            )
        elif model_type == ModelTypes.MULTI_CLASS:
            df["ARIZE_PLACEHOLDER_MAP"] = [
                [
                    {
                        "class_name": "ARIZE_DUMMY",
                        "score": 1.0,
                    }
                ]
            ]
            schema = Schema(
                timestamp_column_name="timestamp",
                prediction_score_column_name="ARIZE_PLACEHOLDER_MAP",
            )
        elif (
            model_type == ModelTypes.NUMERIC
            or model_type == ModelTypes.REGRESSION
        ):
            schema = Schema(
                timestamp_column_name="timestamp",
                prediction_label_column_name="ARIZE_PLACEHOLDER_FLOAT",
                prediction_score_column_name="ARIZE_PLACEHOLDER_FLOAT",
                actual_label_column_name="ARIZE_PLACEHOLDER_FLOAT",
                actual_score_column_name="ARIZE_PLACEHOLDER_FLOAT",
            )
        else:
            schema = Schema(
                timestamp_column_name="timestamp",
                prediction_label_column_name="ARIZE_PLACEHOLDER_STRING",
                prediction_score_column_name="ARIZE_PLACEHOLDER_FLOAT",
                actual_label_column_name="ARIZE_PLACEHOLDER_STRING",
                actual_score_column_name="ARIZE_PLACEHOLDER_FLOAT",
            )

        return self._client.log(
            dataframe=df,
            schema=schema,
            environment=environment,
            model_id=model_id,
            model_type=model_type,
        )
