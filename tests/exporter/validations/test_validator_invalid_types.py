import unittest
from datetime import datetime

from arize.exporter.utils.validation import Validator


class MyTestCase(unittest.TestCase):
    def test_zero_error(self):
        try:
            for key, val in valid_inputs.items():
                Validator.validate_input_type(val[0], key, val[1])
        except TypeError:
            self.fail("validate_input_type raised TypeError unexpectedly")

    def test_invalid_space_id(self):
        space_id = invalid_inputs["space_id"][0]
        valid_type = invalid_inputs["space_id"][1]
        with self.assertRaisesRegex(
            TypeError,
            f"space_id {space_id} is type {type(space_id)}, but must be a str",
        ):
            Validator.validate_input_type(space_id, "space_id", valid_type)

    def test_invalid_model_name(self):
        model_name = invalid_inputs["model_name"][0]
        valid_type = invalid_inputs["model_name"][1]
        with self.assertRaisesRegex(
            TypeError,
            f"model_name {model_name} is type {type(model_name)}, but must be a str",
        ):
            Validator.validate_input_type(model_name, "model_name", valid_type)

    def test_invalid_data_type(self):
        data_type = invalid_inputs["data_type"][0]
        valid_type = invalid_inputs["data_type"][1]
        with self.assertRaisesRegex(
            TypeError,
            f"data_type {data_type} is type {type(data_type)}, but must be a str",
        ):
            Validator.validate_input_type(data_type, "data_type", valid_type)

    def test_invalid_start_or_end_time(self):
        start_time = invalid_inputs["start_time"][0]
        valid_type = invalid_inputs["start_time"][1]
        with self.assertRaisesRegex(
            TypeError,
            f"start_time {start_time} is type {type(start_time)}, but must be a datetime",
        ):
            Validator.validate_input_type(start_time, "start_time", valid_type)

    def test_invalid_path(self):
        path = invalid_inputs["path"][0]
        valid_type = invalid_inputs["path"][1]
        with self.assertRaisesRegex(
            TypeError,
            f"path {path} is type {type(path)}, but must be a str",
        ):
            Validator.validate_input_type(path, "path", valid_type)


valid_inputs = {
    "space_id": ("abc123", str),
    "model_name": ("test_model", str),
    "data_type": ("predictions", str),
    "start_time": (datetime(2023, 4, 1, 0, 0, 0, 0), datetime),
    "end_time": (datetime(2023, 4, 15, 0, 0, 0, 0), datetime),
    "path": ("example.parquet", str),
}

invalid_inputs = {
    "space_id": (123, str),
    "model_name": (123, str),
    "data_type": (0.2, str),
    "start_time": ("2022-10-10", datetime),
    "end_time": ("2022-10-15", datetime),
    "path": (0.2, str),
}

if __name__ == "__main__":
    unittest.main()
