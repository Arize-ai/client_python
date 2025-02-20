import ast

import numpy as np
import pandas as pd

from arize.utils.logging import logger


class WhylabsProfileAdapter:
    """A utility class for generating synthetic data based on WhyLogs profiles.

    This class serves to provide integration support for WhyLabs users to Arize by generating
    synthetic data similar to the sensitive data that customers are ingesting.

    Generates synthetic data that matches the statistical properties of an input DataFrame using
    WhyLogs profiling. It supports various data types including numeric (integral and fractional),
    string, and boolean values, while preserving the original data's distribution characteristics.

    Key Features:
        - Generates synthetic data maintaining statistical properties of the original dataset
        - Handles missing values with appropriate proportions
        - Special handling for zero-dominant numeric distributions
        - Preserves frequency distributions for categorical data

    Example:
        ```python
        import pandas as pd
        from arize.experimental.integrations.whylabs.generator import (
            SyntheticDataGenerator,
        )

        # Existing the WhyLogs integration
        results = why.log(original_df)

        # Generate a WhyLogs profile and convert it into a dataframe for Arize integration
        profile = results.profile()
        profile_view = profile.view()
        profile_df = profile_view.to_pandas()

        # Generate synthetic data from the profile view
        synthetic_df = WhylabsProfileAdapter.generate(
            profile_df,
            num_rows=len(original_df),
            kll_profile_view=profile_view,  # pass in your full profile view
            n_kll_quantiles=500,  # how many quantiles to sample
        )
        ```
    """

    @staticmethod
    def generate(
        profile_df, num_rows, kll_profile_view=None, n_kll_quantiles=200
    ):
        """
        Generate synthetic data based on the profile DataFrame. For numeric columns,
        we will try to sample from a WhyLogs KLL sketch if provided (kll_profile_view),
        else fallback to the existing zero-dominant/quantile-based logic.

        :param profile_df: Pandas DataFrame from profile_view.to_pandas()
        :param num_rows: how many rows to generate
        :param kll_profile_view: (Optional) a whylogs.DatasetProfileView object with KLL data
        :param n_kll_quantiles: how many quantiles to sample if using KLL
        """
        synthetic_df = {}

        for column_name, row in profile_df.iterrows():
            logger.info(f"Processing column: {column_name}")
            # Determine column type
            column_type = None
            if row.get("types/integral", 0) > 0:
                column_type = "integral"
            elif row.get("types/fractional", 0) > 0:
                column_type = "fractional"
            elif row.get("types/string", 0) > 0:
                column_type = "string"
            elif row.get("types/boolean", 0) > 0:
                column_type = "boolean"

            # Handle missing values
            counts_nan = float(row.get("counts/nan", 0))
            counts_null = float(row.get("counts/null", 0))
            counts_total = float(row.get("counts/n", num_rows))
            missing_prob = (
                (counts_nan + counts_null) / counts_total
                if counts_total > 0
                else 0
            )

            if column_type in ["integral", "fractional"]:
                data = WhylabsProfileAdapter._generate_numeric_distribution(
                    row,
                    num_rows,
                    column_type,
                    column_name=column_name,
                    kll_profile_view=kll_profile_view,
                    n_kll_quantiles=n_kll_quantiles,
                )

            elif column_type == "string":
                frequent_items = row.get("frequent_items/frequent_strings", [])
                items = WhylabsProfileAdapter._parse_frequent_items(
                    frequent_items
                )
                if items:
                    values, weights = zip(*items)
                    weights = np.array(weights, dtype=float)
                    weights = weights / weights.sum()
                    data = np.random.choice(values, size=num_rows, p=weights)
                else:
                    cardinality = int(row.get("cardinality/est", 10))
                    unique_values = [f"value_{i}" for i in range(cardinality)]
                    data = np.random.choice(unique_values, size=num_rows)

            elif column_type == "boolean":
                true_count = float(row.get("counts/true", 0))
                total_count = float(row.get("counts/n", num_rows))
                true_prob = true_count / total_count if total_count > 0 else 0.5
                data = np.random.choice(
                    [True, False], size=num_rows, p=[true_prob, 1 - true_prob]
                )

            # Fallback
            else:
                data = np.array([f"{column_name}_{i}" for i in range(num_rows)])

            # Apply missing values
            if missing_prob > 0:
                mask = np.random.random(num_rows) < missing_prob
                data = np.where(mask, None, data)

            synthetic_df[column_name] = data
        logger.info("Done!")
        return pd.DataFrame(synthetic_df)

    ########################################
    # HELPER FOR FREQUENT-ITEM STRINGS
    ########################################
    @staticmethod
    def _parse_frequent_items(frequent_items_str):
        """Parse the frequent items string into a list of tuples (value, est)."""
        if not frequent_items_str or frequent_items_str == "[]":
            return []
        try:
            if isinstance(frequent_items_str, str):
                try:
                    items = ast.literal_eval(frequent_items_str)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Failed to parse frequent items string: {e}")
                    return []
                return [(item["value"], item["est"]) for item in items]
            elif isinstance(frequent_items_str, list):
                return [
                    (str(item.value), item.est) for item in frequent_items_str
                ]
        except (AttributeError, KeyError) as e:
            logger.error(f"Invalid format in frequent items: {e}")
            return []

    ########################################
    # KLL SAMPLING
    ########################################
    @staticmethod
    def _sample_from_kll(kll_sketch, num_rows, n_kll_quantiles=200):
        """
        Given a KLL sketch, sample a continuous distribution of 'num_rows' points
        by drawing random probabilities in [0,1] and inverting them through the
        KLL-based quantiles.

        :param kll_sketch: The KLL object from WhyLogs (dist_metric.kll.value).
        :param num_rows: Number of points to generate.
        :param n_kll_quantiles: Resolution of the KLL sampling. Higher = smoother.
        """
        # 1) Build an array of probabilities
        probs = np.linspace(0, 1, n_kll_quantiles)
        # 2) Get the approximate quantiles from KLL at those probabilities
        quantile_values = kll_sketch.get_quantiles(probs.tolist())
        # 3) Generate uniform random numbers in [0,1] for each row
        random_probs = np.random.rand(num_rows)
        # 4) Interpolate those random probabilities into actual numeric values
        sampled_values = np.interp(random_probs, probs, quantile_values)
        return sampled_values

    ########################################
    # REGULAR NON-ZERO-DOMINANT GENERATION
    ########################################
    @staticmethod
    def _generate_regular_distribution(row, num_rows):
        """Generate regular (non-zero-dominant) numeric distribution."""
        min_val = float(row.get("distribution/min", 0))
        max_val = float(row.get("distribution/max", 100))
        mean = float(row.get("distribution/mean", (min_val + max_val) / 2))
        stddev = float(row.get("distribution/stddev", (max_val - min_val) / 6))

        quantile_points = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        quantile_keys = [
            "distribution/q_01",
            "distribution/q_05",
            "distribution/q_10",
            "distribution/q_25",
            "distribution/median",
            "distribution/q_75",
            "distribution/q_90",
            "distribution/q_95",
            "distribution/q_99",
        ]

        quantiles = []
        values = []
        for prob, key in zip(quantile_points, quantile_keys):
            val = row.get(key)
            if val is not None and not pd.isna(val):
                quantiles.append(prob)
                values.append(float(val))

        if not values:
            # Fallback to normal distribution
            data = np.random.normal(mean, stddev, num_rows)
            return np.clip(data, min_val, max_val)

        # Interpolate random values from these quantiles
        random_probs = np.random.uniform(0, 1, num_rows)
        data = np.interp(random_probs, quantiles, values)
        return data

    ########################################
    # ZERO-DOMINANT OR REGULAR DISTRIBUTION
    ########################################
    @staticmethod
    def _generate_numeric_distribution(
        row,
        num_rows,
        column_type,
        column_name=None,
        kll_profile_view=None,
        n_kll_quantiles=200,
    ):
        """
        Enhanced numeric distribution generator with special handling for zero-dominant
        distributions. If a KLL sketch is provided (kll_profile_view), we instead sample
        directly from that sketch for a continuous distribution.

        :param row: A single row from the profile_df describing numeric stats.
        :param num_rows: How many data points to generate.
        :param column_type: 'integral' or 'fractional'
        :param column_name: Name of the current column (used to look up KLL in the profile).
        :param kll_profile_view: (Optional) A WhyLogs DatasetProfileView to get KLL sketches from.
        :param n_kll_quantiles: Number of quantiles used if we sample from the KLL.
        :return: A NumPy array of generated numeric data.
        """

        # -----------------------------------------------------------
        # 1) If we have a WhyLogs KLL for this column, sample from it
        # -----------------------------------------------------------
        if kll_profile_view is not None and column_name is not None:
            col_view = kll_profile_view.get_column(column_name)
            if col_view is not None:
                dist_metric = col_view._metrics.get("distribution")
                if (
                    dist_metric is not None
                    and getattr(dist_metric.kll, "value", None) is not None
                ):
                    kll_sketch = dist_metric.kll.value
                    data = WhylabsProfileAdapter._sample_from_kll(
                        kll_sketch, num_rows, n_kll_quantiles=n_kll_quantiles
                    )

                    # If integral, just round
                    if column_type == "integral":
                        data = np.round(data).astype(int)
                    return data
        # -----------------------------------------------------------
        # 2) Otherwise, fallback to the “zero-dominant” or “regular” code
        # -----------------------------------------------------------

        min_val = float(row.get("distribution/min", 0))
        max_val = float(row.get("distribution/max", 100))
        median = float(row.get("distribution/median", (min_val + max_val) / 2))
        mean = float(row.get("distribution/mean", median))

        # Get quantiles
        q01 = float(row.get("distribution/q_01", 0))
        q05 = float(row.get("distribution/q_05", 0))
        q10 = float(row.get("distribution/q_10", 0))
        q25 = float(row.get("distribution/q_25", 0))
        q50 = float(row.get("distribution/median", 0))
        q75 = float(row.get("distribution/q_75", 0))
        q90 = float(row.get("distribution/q_90", 0))
        q95 = float(row.get("distribution/q_95", 0))
        q99 = float(row.get("distribution/q_99", 0))

        # Detect zero-dominant distribution
        zero_dominant = False
        zero_proportion = 0.0

        # Case 1: Check if multiple lower quantiles are zero
        zero_quantiles = sum(
            1 for q in [q01, q05, q10, q25, q50, q75, q90, q95, q99] if q == 0
        )
        if zero_quantiles >= 3:  # If at least three quantiles are zero
            zero_dominant = True
            if q99 == 0:
                zero_proportion = 0.99
            elif q95 == 0:
                zero_proportion = 0.95
            elif q90 == 0:
                zero_proportion = 0.90
            elif q75 == 0:
                zero_proportion = 0.75
            elif q50 == 0:
                zero_proportion = 0.50
            elif q25 == 0:
                zero_proportion = 0.25
            elif q10 == 0:
                zero_proportion = 0.10
            elif q05 == 0:
                zero_proportion = 0.05
            elif q01 == 0:
                zero_proportion = 0.01

        # Case 2: Check mean vs max for additional zero detection
        if mean < max_val * 0.01 and max_val > 0:
            zero_dominant = True
            # Estimate zero proportion from mean and max value
            zero_proportion = max(zero_proportion, 1 - (mean / max_val))

        if zero_dominant:
            # Generate zero-dominated distribution
            n_zeros = int(num_rows * zero_proportion)
            n_nonzeros = num_rows - n_zeros
            # Initialize array with zeros
            data = np.zeros(num_rows)

            if n_nonzeros > 0:
                # Generate non-zero values only for the remaining portion
                non_zero_values = []
                non_zero_weights = []
                # Collect non-zero quantiles and their probabilities
                quantile_pairs = [
                    (0.01, q01),
                    (0.05, q05),
                    (0.10, q10),
                    (0.25, q25),
                    (0.50, q50),
                    (0.75, q75),
                    (0.90, q90),
                    (0.95, q95),
                    (0.99, q99),
                    (1.00, max_val),
                ]
                prev_prob = zero_proportion
                for prob, value in quantile_pairs:
                    if value is not None and float(value) > 0:
                        non_zero_values.append(float(value))
                        weight = (prob - prev_prob) / (1 - zero_proportion)
                        non_zero_weights.append(max(0, weight))
                        prev_prob = prob

                if non_zero_values:
                    # Normalize weights
                    non_zero_weights = np.array(non_zero_weights)
                    non_zero_weights = non_zero_weights / non_zero_weights.sum()

                    # Generate non-zero values
                    selected_values = np.random.choice(
                        non_zero_values, size=n_nonzeros, p=non_zero_weights
                    )

                    # Add noise if fractional to prevent exact duplicates
                    if column_type == "fractional":
                        noise = np.random.normal(0, max_val / 1000, n_nonzeros)
                        selected_values = np.clip(
                            selected_values + noise, 0, max_val
                        )

                    # Place non-zeros values randomly
                    non_zero_indices = np.random.choice(
                        num_rows, size=n_nonzeros, replace=False
                    )
                    data[non_zero_indices] = selected_values
        else:
            # Fallback: generate a "regular" distribution
            data = WhylabsProfileAdapter._generate_regular_distribution(
                row, num_rows
            )

        # If integral, round
        if column_type == "integral":
            data = np.round(data).astype(int)

        return data
