from abc import ABC, abstractmethod


class DatasetError(Exception, ABC):
    def __str__(self) -> str:
        return self.error_message()

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def error_message(self) -> str:
        pass


class InvalidSessionError(DatasetError):
    @staticmethod
    def error_message() -> str:
        return (
            "Credentials not provided or invalid. Please pass in the correct api_key when "
            "initiating a new ArizeExportClient. Alternatively, you can set up credentials "
            "in a profile or as an environment variable"
        )


class InvalidConfigFileError(DatasetError):
    @staticmethod
    def error_message() -> str:
        return "Invalid/Misconfigured Configuration File"


class IDColumnUniqueConstraintError(DatasetError):
    @staticmethod
    def error_message() -> str:
        return "'id' column must contain unique values"


class RequiredColumnsError(DatasetError):
    def __init__(self, missing_columns: set) -> None:
        self.missing_columns = missing_columns

    def error_message(self) -> str:
        return f"Missing required columns: {self.missing_columns}"

    def __repr__(self) -> str:
        return f"RequiredColumnsError({self.missing_columns})"
