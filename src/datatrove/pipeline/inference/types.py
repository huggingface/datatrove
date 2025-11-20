from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from datatrove.data import Document


@dataclass
class InferenceResult:
    """
    Successful inference result.

    Attributes:
        text: Generated text from the model
        finish_reason: Reason why generation finished
        usage: Token usage statistics from the model
    """

    text: str
    finish_reason: str
    usage: dict


class InferenceError(Exception):
    """
    Exception raised when document inference processing fails.

    Attributes:
        document: The original document that failed processing
        error: The underlying error that caused the failure
    """

    def __init__(self, document: Document | None, error: str | Exception, payload: dict | None = None):
        self.document = document
        self.error = error
        self.payload = payload
        super().__init__(
            f"Failed to process document {document.id if document is not None else '?'}: {error}. Payload: {payload if payload is not None else '?'}"
        )


class ServerError(Exception):
    """
    Exception raised when the server fails to process a request.
    """

    def __init__(self, error: str | Exception):
        self.error = error
        if not isinstance(error, ServerError):
            error_str = f"Server encountered unrecoverable error: {error}"
        else:
            error_str = str(error)

        super().__init__(error_str)


# Type alias for the generate callback function
# Takes a payload dictionary and returns an awaitable InferenceResult
# May raise InferenceError if the request fails
GenerateFunction = Callable[[dict], Awaitable[InferenceResult]]

# Type alias for rollout function return values
# Rollout functions can return InferenceResult or any JSON-serializable value
RolloutResult = InferenceResult | dict | list | str | float | int | bool | None


class RolloutFunction(Protocol):
    """
    Type for rollout functions that process documents.

    Rollout functions receive:
    - document: The document to process
    - generate: A callback function to send requests to the inference server
    - **kwargs: Arbitrary keyword arguments from shared_context (e.g., process_pool, coding_env, etc.)

    Returns an awaitable that resolves to InferenceResult or a JSON-serializable value.
    """

    def __call__(
        self,
        document: Document,
        generate: GenerateFunction,
        **kwargs: Any,
    ) -> Awaitable[RolloutResult]: ...
