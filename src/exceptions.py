
class ScoringNotFoundError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class TargetVariableNotFoundError(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class WrongEstimatorTypeError(TypeError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class ModelNotSetError(AttributeError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class TRPNotFoundError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class WrongSplittingMode(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class WrongTrainRecordsRetrievalMode(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class WrongSQLStatement(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class MissingDataException(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class NoDataError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class ProjectDBNotFoundError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class ModelBestParametersNotFound(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class RoadCategoryNotFound(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class WrongGraphProcessingBackendError(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)

















