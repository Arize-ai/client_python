import unittest
from datetime import datetime

from arize.exporter.utils.validation import Validator


class MyTestCase(unittest.TestCase):
    def test_valid_data_type(self):
        try:
            Validator.validate_input_value(valid_data_type.upper(), "data_type", data_types)
        except TypeError:
            self.fail("validate_input_type raised TypeError unexpectedly")

    def test_invalid_data_type(self):
        with self.assertRaisesRegex(
            ValueError,
            f"data_type is {invalid_data_type.upper()}, but must be one of PREDICTIONS, "
            f"CONCLUSIONS, EXPLANATIONS, PREPRODUCTION",
        ):
            Validator.validate_input_value(invalid_data_type.upper(), "data_type", data_types)

    def test_invalid_start_end_time(self):
        with self.assertRaisesRegex(
            ValueError,
            "start_time must be before end_time",
        ):
            Validator.validate_start_end_time(start_time, end_time)


valid_data_type = "preproduction"
invalid_data_type = "hello"
data_types = ("PREDICTIONS", "CONCLUSIONS", "EXPLANATIONS", "PREPRODUCTION")
start_time = datetime(2023, 6, 15, 10, 30)
end_time = datetime(2023, 6, 10, 10, 30)

if __name__ == "__main__":
    unittest.main()
