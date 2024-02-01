from enum import Enum, unique


@unique
class StatusCodes(Enum):
    UNSET = 0
    OK = 1
    ERROR = 2

    @classmethod
    def list_codes(cls):
        return [t.name for t in cls]
