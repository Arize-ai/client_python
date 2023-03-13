import json
from collections import namedtuple

import pytest
from arize.utils import utils


def test_response_url():
    input = namedtuple("Response", ["content"])(
        json.dumps(
            {
                "realTimeIngestionUri": (
                    "https://app.dev.arize.com/"
                    "organizations/5/"
                    "spaces/5/"
                    "modelName/"
                    "z-upload-classification-data-with-arize?"
                    "selectedTab=dataIngestion"
                )
            }
        ).encode()
    )
    expected = (
        "https://app.dev.arize.com/"
        "organizations/QWNjb3VudE9yZ2FuaXphdGlvbjo1/"
        "spaces/U3BhY2U6NQ==/models/"
        "modelName/"
        "z-upload-classification-data-with-arize?"
        "selectedTab=dataIngestion"
    )
    assert utils.reconstruct_url(input) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
