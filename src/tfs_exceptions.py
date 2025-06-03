
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


class TargetDataNotAvailableError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)






