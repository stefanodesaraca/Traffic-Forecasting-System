
class ScoringNotFoundError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class TargetVariableNotFoundError(ValueError):
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


class WrongSQLStatementError(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class MissingDataError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class ProjectDBNotFoundError(Exception):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)


class RoadCategoryNotFound(ValueError):
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)
