class EdgeAIError(Exception):
    """Base error for platform failures."""

class ValidationError(EdgeAIError):
    pass

class ArtifactIntegrityError(EdgeAIError):
    pass

class SignatureError(ArtifactIntegrityError):
    pass

class RegistryError(EdgeAIError):
    pass

class DeploymentError(EdgeAIError):
    pass

class RollbackRejected(DeploymentError):
    pass

class HealthGateError(DeploymentError):
    pass

class DeploymentConflict(DeploymentError):
    pass

class QueueFull(EdgeAIError):
    pass

class QueueDropped(QueueFull):
    pass

class DeadlineExceeded(EdgeAIError):
    pass

class ExecutorClosed(EdgeAIError):
    pass

class ExecutionDeadlineExceeded(DeadlineExceeded):
    pass

class BackendProcessError(EdgeAIError):
    pass
