# type: ignore[pb2]
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from whylogs.core import DatasetProfile, DatasetProfileView

from arize.experimental.integrations.whylabs.generator import (
    WhylabsProfileAdapter,
)
from arize.pandas.logger import Client
from arize.utils.types import (
    BaseSchema,
    Environments,
    Metrics,
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
        self._profile_adapter = WhylabsProfileAdapter()

    def log_profile(
        self,
        profile: DatasetProfile,
        schema: BaseSchema,
        environment: Environments,
        model_id: str,
        model_type: ModelTypes,
        metrics_validation: Optional[List[Metrics]] = None,
        model_version: Optional[str] = None,
        batch_id: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        # Synthetic dataframe generation parameters
        kll_profile_view: Optional[DatasetProfileView] = None,
        n_kll_quantiles: Optional[int] = 200,
    ) -> requests.Response:
        """
        Logs a WhyLogs profile to the Arize API.

        The input must be a WhyLogs profile.
        """
        assert isinstance(
            profile, DatasetProfile
        ), f"Expected WhyLogs DatasetProfile, got {type(profile)}"

        synthetic_df = self._generate_synthetic_dataset(
            profile,
            num_rows=len(profile.view().to_pandas()),
            kll_profile_view=kll_profile_view,
            n_kll_quantiles=n_kll_quantiles,
        )

        return self._client.log(
            dataframe=synthetic_df,
            schema=schema,
            environment=environment,
            model_id=model_id,
            model_type=model_type,
            metrics_validation=metrics_validation,
            model_version=model_version,
            batch_id=batch_id,
            sync=sync,
            validate=validate,
            path=path,
            timeout=timeout,
            verbose=verbose,
        )

    def log_dataset_profile(
        self,
        profile: DatasetProfile,
        model_id: str,
        metrics_validation: Optional[List[Metrics]] = None,
        model_version: Optional[str] = None,
        sync: Optional[bool] = False,
        validate: Optional[bool] = True,
        path: Optional[str] = None,
        timeout: Optional[float] = None,
        verbose: Optional[bool] = False,
        # Synthetic dataset generation parameters
        kll_profile_view: Optional[DatasetProfileView] = None,
        n_kll_quantiles: Optional[int] = 200,
        # Schema parameters
        prediction_label_column_name: Optional[str] = None,
        prediction_score_column_name: Optional[str] = None,
        actual_label_column_name: Optional[str] = None,
        actual_score_column_name: Optional[str] = None,
        timestamp_column_name: Optional[str] = None,
    ) -> requests.Response:
        """
        Logs a WhyLogs dataset-based profile to the Arize API.

        The input must be a WhyLogs profile.
        """
        assert isinstance(
            profile, DatasetProfile
        ), f"Expected WhyLogs DatasetProfile, got {type(profile)}"

        synthetic_df = self._generate_synthetic_dataset(
            profile,
            num_rows=len(profile.view().to_pandas()),
            kll_profile_view=kll_profile_view,
            n_kll_quantiles=n_kll_quantiles,
        )

        synthetic_df["ARIZE_PLACEHOLDER_STRING"] = "ARIZE_PLACEHOLDER"
        synthetic_df["ARIZE_PLACEHOLDER_FLOAT"] = float("-inf")

        schema = Schema(
            feature_column_names=synthetic_df.columns.tolist(),
            prediction_label_column_name="ARIZE_PLACEHOLDER_STRING",
            prediction_score_column_name="ARIZE_PLACEHOLDER_FLOAT",
            actual_label_column_name="ARIZE_PLACEHOLDER_STRING",
            actual_score_column_name="ARIZE_PLACEHOLDER_FLOAT",
        )

        if timestamp_column_name:
            schema.timestamp_column_name = timestamp_column_name
        if prediction_score_column_name:
            schema.prediction_score_column_name = prediction_score_column_name
        if prediction_label_column_name:
            schema.prediction_label_column_name = prediction_label_column_name
        if actual_label_column_name:
            schema.actual_label_column_name = actual_label_column_name
        if actual_score_column_name:
            schema.actual_score_column_name = actual_score_column_name

        return self._client.log(
            dataframe=synthetic_df,
            schema=schema,
            environment=Environments.PRODUCTION,  # Hardcoded for production env
            model_id=model_id,
            model_type=ModelTypes.BINARY_CLASSIFICATION,  # Hardcoded as classification model
            metrics_validation=metrics_validation,
            model_version=model_version,
            sync=sync,
            validate=validate,
            path=path,
            timeout=timeout,
            verbose=verbose,
        )

    def _generate_synthetic_dataset(
        self,
        profile: DatasetProfile,
        num_rows: int,
        kll_profile_view: Optional[DatasetProfileView] = None,
        n_kll_quantiles: Optional[int] = 200,
    ) -> pd.DataFrame:
        profile_df = profile.view().to_pandas()
        return self._profile_adapter.generate(
            profile_df,
            num_rows=num_rows,
            kll_profile_view=kll_profile_view,
            n_kll_quantiles=n_kll_quantiles,
        )
