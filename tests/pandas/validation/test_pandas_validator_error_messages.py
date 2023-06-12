import arize.pandas.validation.errors as err
import pytest
from arize.utils.types import Environments, ModelTypes

# ----------------
# Parameter checks
# ----------------


def test_missing_columns():
    err_msg = str(err.MissingColumns(["genotype", "phenotype"]))
    assert "genotype" in err_msg
    assert "phenotype" in err_msg


def test_invalid_model_type():
    err_msg = str(err.InvalidModelType())
    assert all(mt.name in err_msg for mt in ModelTypes)


def test_invalid_environment():
    err_msg = str(err.InvalidEnvironment())
    assert all(env.name in err_msg for env in Environments)


# -----------
# Type checks
# -----------


def test_Invalid_type():
    err_msg = str(err.InvalidType("123", ["456", "789"], "112"))
    assert "123" in err_msg
    assert "456" in err_msg
    assert "789" in err_msg
    assert "112" in err_msg


def test_invalid_type_features():
    err_msg = str(
        err.InvalidTypeFeatures(["genotype", "phenotype"], ["Triceratops", "Archaeopteryx"])
    )
    assert "genotype" in err_msg
    assert "phenotype" in err_msg
    assert "Triceratops" in err_msg
    assert "Archaeopteryx" in err_msg


def test_invalid_type_tags():
    err_msg = str(err.InvalidTypeTags(["genotype", "phenotype"], ["Triceratops", "Archaeopteryx"]))
    assert "genotype" in err_msg
    assert "phenotype" in err_msg
    assert "Triceratops" in err_msg
    assert "Archaeopteryx" in err_msg


def test_invalid_type_shap_values():
    err_msg = str(
        err.InvalidTypeShapValues(["genotype", "phenotype"], ["Triceratops", "Archaeopteryx"])
    )
    assert "genotype" in err_msg
    assert "phenotype" in err_msg
    assert "Triceratops" in err_msg
    assert "Archaeopteryx" in err_msg


# ------------
# Value checks
# ------------


def test_invalid_timestamp():
    err_msg = str(err.InvalidValueTimestamp("Spinosaurus"))
    assert "Spinosaurus" in err_msg


def test_invalid_missing_value():
    err_msg = str(err.InvalidValueMissingValue("Stegosaurus", "missing"))
    assert "Stegosaurus" in err_msg
    assert "missing" in err_msg


def test_invalid_infinite_value():
    err_msg = str(err.InvalidValueMissingValue("Stegosaurus", "infinite"))
    assert "Stegosaurus" in err_msg
    assert "infinite" in err_msg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
