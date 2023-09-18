from typing import Optional, Union

from .constants import API_KEY_ENVVAR_NAME, SPACE_KEY_ENVVAR_NAME


class AuthError(Exception):
    def __init__(self, api_key: Optional[str], space_key: Optional[str]) -> None:
        self.missing_api_key = api_key is None
        self.missing_space_key = space_key is None

    def __repr__(self) -> str:
        return "Invalid_Arize_Client_Authentication"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        missing_list = ["api_key"] if self.missing_api_key else []
        if self.missing_space_key:
            missing_list.append("space_key")

        return (
            "Arize Client could not obtain credentials. You can pass your api_key and space_key "
            "directly to the Arize Client, or you can set environment variables which will be read if the "
            "keys are not directly passed. "
            "To set the environment variables use the following variable names: \n"
            f" - {API_KEY_ENVVAR_NAME} for the api key\n"
            f" - {SPACE_KEY_ENVVAR_NAME} for the space key\n"
            f"Missing: {missing_list}"
        )


class InvalidStringLength(Exception):
    def __init__(self, name: str, min_length: int, max_length: int) -> None:
        self.name = name
        self.min_length = min_length
        self.max_length = max_length

    def __repr__(self) -> str:
        return "Invalid_String_Length"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        return f"{self.name} must be of length between {self.min_length} and {self.max_length} characters."


class InvalidFieldType(Exception):
    def __init__(self, name: str, value: Union[bool, int, float, str], correct_type: int) -> None:
        self.name = name
        self.value = value
        self.correct_type = correct_type

    def __repr__(self) -> str:
        return "Invalid_Field_Type"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        return f"{self.name} {self.value} is of type {type(self.value)}, but must be of {self.correct_type}"


class InvalidValueType(Exception):
    def __init__(
        self,
        value_name: str,
        value: Union[bool, int, float, str],
        correct_type: int,
    ) -> None:
        self.value_name = value_name
        self.value = value
        self.correct_type = correct_type

    def __repr__(self) -> str:
        return "Invalid_Value_Type"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        return (
            f"{self.value_name} with value {self.value} is of type {type(self.value).__name__}, "
            f"but expected {self.correct_type}"
        )
