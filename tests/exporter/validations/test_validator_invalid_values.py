import unittest

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


valid_data_type = "preproduction"
invalid_data_type = "hello"
data_types = ("PREDICTIONS", "CONCLUSIONS", "EXPLANATIONS", "PREPRODUCTION")

if __name__ == "__main__":
    unittest.main()
