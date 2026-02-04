"""Arize SDK for model observability and LLM tracing."""

import logging
from collections.abc import Mapping
from typing import Literal, cast

from arize._generated.api_client import models
from arize.client import ArizeClient
from arize.config import SDKConfiguration
from arize.regions import Region

# Attach a NullHandler by default in the top-level package
# so that if no configuration is installed, nothing explodes.
logging.getLogger("arize").addHandler(logging.NullHandler())

# Opt-in env-based logging
try:
    from .logging import auto_configure_from_env

    auto_configure_from_env()
except Exception:  # noqa: S110
    # Intentionally silent: logging configuration is optional and should never
    # prevent SDK initialization. Users can configure logging explicitly if needed.
    pass

__all__ = [
    "ArizeClient",
    "Region",
    "SDKConfiguration",
]


def make_to_df(field_name: str) -> object:
    def to_df(
        self: object,
        by_alias: bool = False,
        exclude_none: str | bool = False,
        json_normalize: bool = False,
        convert_dtypes: bool = True,
    ) -> object:
        """Convert a list of objects to a :class:`pandas.DataFrame`.

        Behavior:
          - If an item is a Pydantic v2 model, use `.model_dump(by_alias=...)`.
          - If an item is a mapping (dict-like), use it as-is.
          - Otherwise, raise a ValueError (unsupported row type).

        Args:
            self (object): The object instance containing the field to convert.
            by_alias (bool): Use field aliases when dumping Pydantic models.
            exclude_none (str | bool): Control None/NaN column dropping.
                - False: keep Nones as-is
                - "all": drop columns where all values are None/NaN
                - "any": drop columns where any value is None/NaN
                - True: alias for "all"
            json_normalize (bool): If True, flatten nested dicts via `pandas.json_normalize`.
            convert_dtypes (bool): If True, call `DataFrame.convert_dtypes()` at the end.

        Returns:
            pandas.DataFrame: The converted DataFrame.
        """
        import pandas as pd

        items = getattr(self, field_name, []) or []

        rows = []
        for it in items:
            if hasattr(it, "model_dump"):  # Pydantic v2 object
                rows.append(it.model_dump(by_alias=by_alias))

            elif isinstance(it, Mapping):  # Plain mapping
                rows.append(it)
            else:
                raise ValueError(
                    f"Cannot convert item of type {type(it)} to DataFrame row"
                )

        df = (
            pd.json_normalize(rows, sep=".")
            if json_normalize
            else pd.DataFrame(rows)
        )

        # Drop None/NaN columns if requested
        if exclude_none in ("any", "all", True):
            drop_how: Literal["any", "all"] = (
                "all"
                if exclude_none is True
                else cast("Literal['any', 'all']", exclude_none)
            )
            df.dropna(axis=1, how=drop_how, inplace=True)

        if convert_dtypes:
            df = df.convert_dtypes()
        return df

    return to_df


# Monkey-patch convenience methods onto generated response models
# Type ignore comments needed: mypy can't verify runtime attribute additions
models.DatasetsList200Response.to_df = make_to_df("datasets")  # type: ignore[attr-defined]
models.DatasetsExamplesList200Response.to_df = make_to_df("examples")  # type: ignore[attr-defined]
models.ExperimentsList200Response.to_df = make_to_df("experiments")  # type: ignore[attr-defined]
models.ExperimentsRunsList200Response.to_df = make_to_df("experiment_runs")  # type: ignore[attr-defined]
models.ProjectsList200Response.to_df = make_to_df("projects")  # type: ignore[attr-defined]
