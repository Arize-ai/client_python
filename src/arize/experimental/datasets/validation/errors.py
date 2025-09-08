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


class EmptyDatasetError(DatasetError):
    @staticmethod
    def error_message() -> str:
        return "DataFrame must have at least one row in it."


class InvalidChunkSizeError(DatasetError):
    def __init__(self, chunk_size: int, max_chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.max_chunk_size = max_chunk_size

    def error_message(self) -> str:
        return f"Invalid chunk size: {self.chunk_size}. Must be between 0 and {self.max_chunk_size}"

    def __repr__(self) -> str:
        return f"InvalidChunkSizeError({self.chunk_size})"
