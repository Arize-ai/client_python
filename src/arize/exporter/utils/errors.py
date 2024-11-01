from abc import ABC, abstractmethod


class ExportingError(Exception, ABC):
    def __str__(self) -> str:
        return self.error_message()

    @abstractmethod
    def error_message(self) -> str:
        pass


class InvalidSessionError(ExportingError):
    @staticmethod
    def error_message() -> str:
        return (
            "Credentials not provided or invalid. Please pass in the correct api_key when "
            "initiating a new ArizeExportClient. Alternatively, you can set up credentials "
            "in a profile or as an environment variable"
        )


class InvalidConfigFileError(Exception):
    @staticmethod
    def error_message() -> str:
        return "WENDY TO WRITE APPROPRIATE ERROR MESSAGE"
