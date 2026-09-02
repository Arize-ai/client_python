"""Upload dataset examples from a parquet file to Arize via Flight RPC.

Example::

    ARIZE_API_KEY_US=your-api-key \
        uv run --project sdk/python/arize/v8 \
        python sdk/python/arize/v8/tests/manual/test_create_member_llm_dataset.py \
        --file member_llm_gts_119k.parquet --env prod --iteration 1

    ARIZE_API_KEY_DEV=your-api-key \
        uv run --project sdk/python/arize/v8 \
        python sdk/python/arize/v8/tests/manual/test_create_member_llm_dataset.py \
        --file member_llm_gts_119k.parquet --env dev --iteration 1
"""

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd

from arize import ArizeClient

DEV_CLIENT_KWARGS = {
    "api_host": "devr.arize.com",
    "otlp_host": "devotlp.arize.com",
    "flight_host": "devx.arize.com",
    "flight_port": 443,
}
SPACE_IDS = {
    "prod": "YOUR-PROD-SPACE-ID-HERE",
    "dev": "YOUR-DEV-SPACE-ID-HERE",
}


def dataset_name(run_date: date, iteration: int) -> str:
    """Return the dataset name for a run date and iteration."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return f"test-kiko-{run_date:%Y-%m-%d}-iter-{iteration}"


def client_kwargs_for_environment(environment: str) -> dict[str, str | int]:
    """Return Arize client configuration for the selected environment."""
    api_key_name = (
        "ARIZE_API_KEY_DEV" if environment == "dev" else "ARIZE_API_KEY_US"
    )
    api_key = os.environ.get(api_key_name)
    if not api_key:
        raise ValueError(f"{api_key_name} environment variable is required")

    client_kwargs: dict[str, str | int] = {
        "api_key": api_key,
        "pyarrow_max_chunksize": 3_000,
    }
    if environment == "dev":
        client_kwargs.update(DEV_CLIENT_KWARGS)
    return client_kwargs


def main(argv: list[str] | None = None, *, run_date: date | None = None) -> int:
    """Read a parquet file and upload it as an Arize dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--env", choices=("prod", "dev"), required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="upload only the first N rows (for testing chunked upload)",
    )
    args = parser.parse_args(argv)
    if not args.file.is_file():
        raise FileNotFoundError(f"parquet file not found: {args.file}")

    name = dataset_name(run_date or date.today(), args.iteration)  # noqa: DTZ011
    print(f"DEBUGPRINT: test_create_dataset_via_flight.py:68: name={name}")

    client = ArizeClient(**client_kwargs_for_environment(args.env))
    sdk_config = client.sdk_config
    print(
        f"DEBUGPRINT: test_create_dataset_via_flight.py:72: sdk_config={sdk_config}"
    )

    space_id = SPACE_IDS[args.env]
    space = client.spaces.get(space=space_id)
    print(f"DEBUGPRINT: test_create_dataset_via_flight.py:77: space={space}")

    data = pd.read_parquet(args.file)
    if args.limit is not None:
        data = data.head(args.limit)
    data.drop(columns=["created_at", "updated_at"], inplace=True)
    print(f"DEBUGPRINT: loaded {len(data)} rows from {args.file}")
    result = client.datasets.create(
        name=name,
        space=space_id,
        examples=data,
    )
    print(result.id)
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
