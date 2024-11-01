import random
import string

import numpy as np
import pandas as pd
import pytest

from arize.utils.constants import MAX_RAW_DATA_CHARACTERS
from arize.utils.types import Embedding


def random_string(N: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


long_raw_data_string = random_string(MAX_RAW_DATA_CHARACTERS)
long_raw_data_token_array = [random_string(7) for _ in range(11000)]
input_embeddings = {
    "correct:complete:list_vector": Embedding(
        vector=[1.0, 2, 3],
        data="this is a test sentence",
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "correct:complete:ndarray_vector+list_data": Embedding(
        vector=np.array([1.0, 2, 3]),
        data=["This", "is", "a", "test", "token", "array"],
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "correct:complete:pdSeries_vector+ndarray_data": Embedding(
        vector=pd.Series([1.0, 2, 3]),
        data=np.array(["This", "is", "a", "test", "token", "array"]),
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "correct:complete:ndarray_vector+pdSeries_data": Embedding(
        vector=np.array([1.0, 2, 3]),
        data=pd.Series(["This", "is", "a", "test", "token", "array"]),
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "correct:missing:data": Embedding(
        vector=np.array([1.0, 2, 3]),
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "correct:missing:link_to_data": Embedding(
        vector=pd.Series([1.0, 2, 3]),
        data=["This", "is", "a", "test", "token", "array"],
    ),
    "correct:empty_vector": Embedding(
        vector=np.array([]),
        data=["This", "is", "a", "test", "token", "array"],
    ),
    "wrong_type:vector": Embedding(
        vector=pd.DataFrame([1.0, 2, 3]),
        data=2,
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "wrong_type:data_num": Embedding(
        vector=pd.Series([1.0, 2, 3]),
        data=2,
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "wrong_type:data_dataframe": Embedding(
        vector=pd.Series([1.0, 2, 3]),
        data=pd.DataFrame(["This", "is", "a", "test", "token", "array"]),
        link_to_data="https://my-bucket.s3.us-west-2.amazonaws.com/puppy.png",
    ),
    "wrong_type:link_to_data": Embedding(
        vector=np.array([1.0, 2, 3]),
        data=["This", "is", "a", "test", "token", "array"],
        link_to_data=True,
    ),
    "wrong_value:size_1_vector": Embedding(
        vector=np.array([1.0]),
        data=["This", "is", "a", "test", "token", "array"],
    ),
    "wrong_value:raw_data_string_too_long": Embedding(
        vector=pd.Series([1.0, 2, 3]),
        data=long_raw_data_string,
    ),
    "wrong_value:raw_data_token_array_too_long": Embedding(
        vector=pd.Series([1.0, 2, 3]),
        data=long_raw_data_token_array,
    ),
}


def test_correct_embeddings():
    keys = [key for key in input_embeddings if "correct:" in key]
    assert len(keys) > 0, "Test configuration error: keys must not be empty"

    for key in keys:
        embedding = input_embeddings[key]
        try:
            embedding.validate(key)
        except Exception as err:
            raise AssertionError(
                f"Correct embeddings should give no errors. Failing key = {key:s}. "
                f"Error = {err}"
            ) from None


def test_wrong_value_fields():
    keys = [key for key in input_embeddings if "wrong_value:" in key]
    assert len(keys) > 0, "Test configuration error: keys must not be empty"

    for key in keys:
        embedding = input_embeddings[key]
        try:
            embedding.validate(key)
        except Exception as err:
            assert isinstance(
                err, ValueError
            ), "Wrong field values should raise value errors"


def test_wrong_type_fields():
    keys = [key for key in input_embeddings if "wrong_type:" in key]
    assert len(keys) > 0, "Test configuration error: keys must not be empty"

    for key in keys:
        embedding = input_embeddings[key]
        try:
            embedding.validate(key)
        except Exception as err:
            assert isinstance(
                err, TypeError
            ), "Wrong field types should raise type errors"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
