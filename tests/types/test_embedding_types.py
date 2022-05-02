import pytest
from arize.utils.types import Embedding, EmbeddingColumnNames
from arize import public_pb2 as pb
import numpy as np
import pandas as pd


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
    "wrong_value:empty_vector": Embedding(
        vector=np.array([]),
        data=["This", "is", "a", "test", "token", "array"],
    ),
}


def test_correct_embeddings():
    keys = [key for key in input_embeddings.keys() if "correct:" in key]
    assert len(keys) > 0, f"Test configuration error: keys must not be empty"

    for key in keys:
        embedding = input_embeddings[key]
        try:
            Embedding.validate_embedding_object(key, embedding)
        except Exception as err:
            assert False, f"Correct embeddings should give no errors. Failing key = {key:s}"


def test_empty_vector():
    keys = [key for key in input_embeddings.keys() if "wrong_value:" in key]
    assert len(keys) > 0, f"Test configuration error: keys must not be empty"

    for key in keys:
        embedding = input_embeddings[key]
        try:
            Embedding.validate_embedding_object(key, embedding)
        except Exception as err:
            assert type(err) == ValueError, "Wrong field values should raise value errors"

def test_wrong_type_fields():
    keys = [key for key in input_embeddings.keys() if "wrong_type:" in key]
    assert len(keys) > 0, f"Test configuration error: keys must not be empty"

    for key in keys:
        embedding = input_embeddings[key]
        try:
            Embedding.validate_embedding_object(key, embedding)
        except Exception as err:
            assert type(err) == TypeError, "Wrong field types should raise type errors"

def test_convert_schema_to_proto():
    schemaEmbeddingColumnNames = [
        EmbeddingColumnNames(
            vector_column_name="image",  # Will be name of embedding feature in the app
            link_to_data_column_name="image_link",
        ),
        EmbeddingColumnNames(
            vector_column_name="sentence",  # Will be name of embedding feature in the app
            data_column_name="sentence_data",
        ),
        EmbeddingColumnNames(
            vector_column_name="token_array",  # Will be name of embedding feature in the app
            data_column_name="token_array_data",
        ),
    ]
    protoEmbeddingColumnNames = [
        pb.Schema.EmbeddingColumnNames(
            vector_column_name = "image",
            link_to_data_column_name = "image_link"
        ),
        pb.Schema.EmbeddingColumnNames(
            vector_column_name = "sentence",
            data_column_name = "sentence_data"
        ),
        pb.Schema.EmbeddingColumnNames(
            vector_column_name = "token_array",
            data_column_name = "token_array_data"
        ),
    ]
    for embColNames, expected_proto in zip(schemaEmbeddingColumnNames, protoEmbeddingColumnNames):
        proto = EmbeddingColumnNames.convert_to_proto(embColNames)
        assert expected_proto == proto

       
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
