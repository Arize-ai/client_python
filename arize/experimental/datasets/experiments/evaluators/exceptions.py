class ArizeException(Exception):
    pass


class ArizeContextLimitExceeded(ArizeException):
    pass


class ArizeTemplateMappingError(ArizeException):
    pass
