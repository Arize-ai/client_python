from typing import Any, Optional, Union

from .constants import API_KEY_ENVVAR_NAME, SPACE_ID_ENVVAR_NAME


class AuthError(Exception):
    def __init__(
        self,
        api_key: Optional[str] = None,
        space_key: Optional[str] = None,
        space_id: Optional[str] = None,
    ) -> None:
        self.missing_api_key = api_key is None
        self.missing_space_key = space_key is None
        self.missing_space_id = space_id is None

    def __repr__(self) -> str:
        return "Invalid_Arize_Client_Authentication"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        missing_list = ["api_key"] if self.missing_api_key else []
        # prompt the user to set space_id only because space_key is getting deprecated
        if self.missing_space_key and self.missing_space_id:
            missing_list.append("space_id")

        return (
            "Arize Client could not obtain credentials."
            "You must pass your api_key and space_id directly to the Arize Client, "
            "or you can set environment variables which will be read if the "
            "keys are not directly passed. "
            "To set the environment variables use the following variable names: \n"
            f" - {API_KEY_ENVVAR_NAME} for the api key\n"
            f" - {SPACE_ID_ENVVAR_NAME} for the space ID\n"
            f"Missing: {missing_list}"
        )


class InvalidCertificateFile(Exception):
    def __init__(self, path_to_certificate: str) -> None:
        self.path_to_certificate = path_to_certificate

    def __repr__(self) -> str:
        return "Invalid_Certificate_File"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        return f"Arize Client could not read certificate file: {self.path_to_certificate}"


class InvalidTypeAuthKey(Exception):
    def __init__(
        self,
        api_key: Optional[Any] = None,
        space_key: Optional[Any] = None,
        space_id: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key
        self.space_key = space_key
        self.space_id = space_id

    def __repr__(self) -> str:
        return "Invalid_Type_Arize_Client_Authentication"

    def __str__(self) -> str:
        return self.error_message()

    def error_message(self) -> str:
        bad_keys_types = {}
        for name, key in [
            ("api_key", self.api_key),
            ("space_key", self.space_key),
            ("space_id", self.space_id),
        ]:
            if key is not None and not isinstance(key, str):
                bad_keys_types[name] = type(key)

        return (
            "Arize Client could not obtain credentials because "
            f"{', '.join(bad_keys_types)} must be string type. "
            "Got: "
            f"{', '.join([f'{key} as {key_type.__name__}' for key, key_type in bad_keys_types.items()])} "
            "instead."
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
