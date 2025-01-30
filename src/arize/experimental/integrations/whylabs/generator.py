import ast

import numpy as np
import pandas as pd
import whylogs as why

from arize.utils.logging import logger


class SyntheticDataGenerator:
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

        # Create original DataFrame
        df = pd.DataFrame(...)

        # Generate synthetic data
        synthetic_df = SyntheticDataGenerator.generate(df)
        ```
    """

    @staticmethod
    def generate(df: pd.DataFrame):
        """Generate synthetic data based on a pandas DataFrame using WhyLogs profiling.

        Requires a WhyLogs API key to be set in the environment variable WHYLABS_API_KEY
        and Org ID to be set in the environment variable WHYLABS_DEFAULT_ORG_ID.

        Args:
            df (pd.DataFrame): Input DataFrame to generate synthetic data from.

        Returns:
            pd.DataFrame: A synthetic DataFrame with the same structure and statistical properties
                         as the input DataFrame.
        """
        results = why.log(df)
        results.writer("whylabs").write()

        # Generate a WhyLogs profile and convert it into a pandas DataFrame for analysis
        profile_df = results.profile().view().to_pandas()
        logger.info(f"\nWhyLogs Profile:\n {profile_df}\n")

        return SyntheticDataGenerator._generate_synthetic_data_from_profile(
            profile_df, num_rows=len(df)
        )

    @staticmethod
    def _generate_synthetic_data_from_profile(profile_df, num_rows):
        """Generate synthetic data based on a WhyLogs profile DataFrame.

        Args:
            profile_df (pd.DataFrame): WhyLogs profile DataFrame containing statistical information.
            num_rows (int): Number of synthetic rows to generate.

        Returns:
            pd.DataFrame: Synthetic DataFrame with generated data matching the profile's properties.
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
                data = SyntheticDataGenerator._generate_numeric_distribution(
                    row, num_rows, column_type
                )

            elif column_type == "string":
                frequent_items = row.get("frequent_items/frequent_strings", [])
                items = SyntheticDataGenerator._parse_frequent_items(
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

            else:
                data = np.array([f"{column_name}_{i}" for i in range(num_rows)])

            # Apply missing values
            if missing_prob > 0:
                mask = np.random.random(num_rows) < missing_prob
                data = np.where(mask, None, data)

            synthetic_df[column_name] = data

        logger.info("Done!")
        return pd.DataFrame(synthetic_df)

    @staticmethod
    def _parse_frequent_items(frequent_items_str):
        """Parse the frequent items string from WhyLogs profile into value-frequency pairs.

        Args:
            frequent_items_str (Union[str, list]): String representation of frequent items or list
                of frequent item objects from WhyLogs profile.

        Returns:
            list[tuple]: List of tuples containing (value, frequency) pairs. Returns empty list
                if parsing fails or input is invalid.
        """
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

    @staticmethod
    def _generate_numeric_distribution(row, num_rows, column_type):
        """Generate synthetic numeric data with special handling for zero-dominant distributions.

        Args:
            row (pd.Series): Row from WhyLogs profile containing distribution statistics.
            num_rows (int): Number of synthetic values to generate.
            column_type (str): Type of numeric data ('integral' or 'fractional').

        Returns:
            np.ndarray: Array of synthetic numeric values matching the distribution properties.
        """
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

                    # Add some noise to prevent exact duplicates
                    if column_type == "fractional":
                        noise = np.random.normal(0, max_val / 1000, n_nonzeros)
                        selected_values = np.clip(
                            selected_values + noise, 0, max_val
                        )

                    # Place non-zero values randomly
                    non_zero_indices = np.random.choice(
                        num_rows, size=n_nonzeros, replace=False
                    )
                    data[non_zero_indices] = selected_values
        else:
            # Use original distribution generation for non-zero-dominant cases
            data = SyntheticDataGenerator._generate_regular_distribution(
                row, num_rows, column_type
            )

        # Handle integral types
        if column_type == "integral":
            data = np.round(data).astype(int)

        return data

    @staticmethod
    def _generate_regular_distribution(row, num_rows, column_type):
        """Generate synthetic numeric data for non-zero-dominant distributions.

        Args:
            row (pd.Series): Row from WhyLogs profile containing distribution statistics.
            num_rows (int): Number of synthetic values to generate.
            column_type (str): Type of numeric data ('integral' or 'fractional').

        Returns:
            np.ndarray: Array of synthetic numeric values following the specified distribution.
        """
        min_val = float(row.get("distribution/min", 0))
        max_val = float(row.get("distribution/max", 100))
        mean = float(row.get("distribution/mean", (min_val + max_val) / 2))
        stddev = float(row.get("distribution/stddev", (max_val - min_val) / 6))

        # Collect quantiles
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

        # Add available quantiles
        for prob, key in zip(quantile_points, quantile_keys):
            val = row.get(key)
            if val is not None and not pd.isna(val):
                quantiles.append(prob)
                values.append(float(val))

        if not values:
            # Fallback to normal distribution if no quantiles available
            data = np.random.normal(mean, stddev, num_rows)
            return np.clip(data, min_val, max_val)

        # Generate distribution based on quantiles
        random_probs = np.random.uniform(0, 1, num_rows)
        data = np.interp(random_probs, quantiles, values)

        return data
