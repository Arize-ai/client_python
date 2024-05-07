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
                    "organizations/test-hmac-org/"
                    "spaces/test-hmac-space/"
                    "models/modelName/"
                    "z-upload-classification-data-with-arize?"
                    "selectedTab=overview"
                )
            }
        ).encode()
    )
    expected = (
        "https://app.dev.arize.com/"
        "organizations/test-hmac-org/"
        "spaces/test-hmac-space/models/"
        "modelName/"
        "z-upload-classification-data-with-arize?"
        "selectedTab=overview"
    )
    assert utils.reconstruct_url(input) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
