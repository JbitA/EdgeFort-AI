from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import hmac
import math

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from .engine import InferenceEngine
from .errors import (
    DeadlineExceeded,
    ExecutionDeadlineExceeded,
    EdgeAIError,
    ExecutorClosed,
    QueueFull,
    ValidationError,
)
from .runtime import BoundedInferenceExecutor, ExecutionMode, OverflowPolicy


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: list[list[float]] = Field(min_length=1, max_length=4096)


class PredictResponse(BaseModel):
    predictions: list[int]
    logits: list[list[float]]
    version: str


class _PayloadTooLarge(Exception):
    pass


class _InvalidBodyFraming(Exception):
    pass


class APIKeyMiddleware:
    """Reject protected requests from raw headers before body/schema allocation."""

    def __init__(self, app, api_key: str, development: bool):
        self.app = app
        self.expected = api_key.encode("utf-8") if api_key else b""
        self.development = development
        self.protected = {"/readyz", "/metrics", "/v1/predict"}

    async def __call__(self, scope, receive, send):
        if self.development or scope["type"] != "http" or scope.get("path") not in self.protected:
            await self.app(scope, receive, send)
            return
        provided = [
            value for key, value in scope.get("headers", []) if key.lower() == b"x-api-key"
        ]
        if len(provided) != 1 or not hmac.compare_digest(provided[0], self.expected):
            response = JSONResponse({"detail": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class RequestConcurrencyLimitMiddleware:
    """Bound concurrent prediction requests before their bodies are consumed."""

    def __init__(self, app, max_inflight: int):
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        self.app = app
        self.max_inflight = max_inflight
        self._guard = asyncio.Lock()
        self._inflight = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/predict":
            await self.app(scope, receive, send)
            return
        async with self._guard:
            if self._inflight >= self.max_inflight:
                response = JSONResponse(
                    {"detail": "too many concurrent inference requests"},
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
                await response(scope, receive, send)
                return
            self._inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            async with self._guard:
                self._inflight -= 1


class RequestBodyLimitMiddleware:
    """ASGI request-body limiter that also covers chunked transfer encoding."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        header_items = [(key.lower(), value) for key, value in scope.get("headers", [])]
        lengths = [value for key, value in header_items if key == b"content-length"]
        transfer_encodings = [value for key, value in header_items if key == b"transfer-encoding"]
        if len(lengths) > 1 or len(transfer_encodings) > 1 or (lengths and transfer_encodings):
            response = JSONResponse({"detail": "ambiguous request body framing"}, status_code=400)
            await response(scope, receive, send)
            return

        content_length = None
        if lengths:
            raw_length = lengths[0]
            if not raw_length or not raw_length.isdigit():
                response = JSONResponse({"detail": "invalid content-length"}, status_code=400)
                await response(scope, receive, send)
                return
            try:
                content_length = int(raw_length)
            except ValueError:
                response = JSONResponse({"detail": "invalid content-length"}, status_code=400)
                await response(scope, receive, send)
                return
            if content_length > self.max_bytes:
                response = JSONResponse({"detail": "request body too large"}, status_code=413)
                await response(scope, receive, send)
                return

        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _PayloadTooLarge
                if content_length is not None and seen > content_length:
                    raise _InvalidBodyFraming
                if not message.get("more_body", False) and content_length is not None and seen != content_length:
                    raise _InvalidBodyFraming
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _PayloadTooLarge:
            response = JSONResponse({"detail": "request body too large"}, status_code=413)
            await response(scope, receive, send)
        except _InvalidBodyFraming:
            response = JSONResponse({"detail": "invalid request body framing"}, status_code=400)
            await response(scope, receive, send)


def create_app(
    engine: InferenceEngine,
    api_key: str | None = None,
    development: bool = False,
    max_request_bytes: int = 8 * 1024 * 1024,
    *,
    queue_capacity: int = 64,
    queue_policy: OverflowPolicy | str = OverflowPolicy.REJECT,
    inference_workers: int = 1,
    queue_deadline_s: float | None = 1.0,
    execution_mode: ExecutionMode | str = ExecutionMode.THREAD,
    execution_timeout_s: float | None = None,
    process_control_timeout_s: float = 5.0,
    process_memory_limit_mb: int | None = None,
    process_nofile_limit: int = 64,
    process_restart_limit: int = 8,
    process_restart_window_s: float = 60.0,
    shutdown_timeout_s: float = 5.0,
) -> FastAPI:
    if not development and not api_key:
        raise RuntimeError("production service requires API key")
    if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int) or max_request_bytes < 1024:
        raise ValueError("max_request_bytes must be an integer >= 1024")
    if queue_deadline_s is not None and (
        not isinstance(queue_deadline_s, (int, float))
        or isinstance(queue_deadline_s, bool)
        or not math.isfinite(queue_deadline_s)
        or queue_deadline_s < 0
    ):
        raise ValueError("queue_deadline_s must be finite and >= 0")
    try:
        resolved_execution_mode = ExecutionMode(execution_mode)
    except (TypeError, ValueError) as e:
        raise ValueError("invalid execution_mode") from e
    if execution_timeout_s is not None and (
        isinstance(execution_timeout_s, bool)
        or not isinstance(execution_timeout_s, (int, float))
        or not math.isfinite(execution_timeout_s)
        or execution_timeout_s <= 0
    ):
        raise ValueError("execution_timeout_s must be finite and > 0")
    if resolved_execution_mode is ExecutionMode.PROCESS and execution_timeout_s is None:
        raise ValueError("process execution mode requires execution_timeout_s")
    if resolved_execution_mode is ExecutionMode.THREAD and execution_timeout_s is not None:
        raise ValueError("execution_timeout_s requires process execution mode")
    if (
        isinstance(process_control_timeout_s, bool)
        or not isinstance(process_control_timeout_s, (int, float))
        or not math.isfinite(process_control_timeout_s)
        or process_control_timeout_s <= 0
    ):
        raise ValueError("process_control_timeout_s must be finite and > 0")
    if process_memory_limit_mb is not None and (
        isinstance(process_memory_limit_mb, bool)
        or not isinstance(process_memory_limit_mb, int)
        or process_memory_limit_mb < 64
        or process_memory_limit_mb > 1_048_576
    ):
        raise ValueError("process_memory_limit_mb must be an integer in [64,1048576]")
    if resolved_execution_mode is ExecutionMode.THREAD and process_memory_limit_mb is not None:
        raise ValueError("process_memory_limit_mb requires process execution mode")
    if (
        isinstance(process_nofile_limit, bool)
        or not isinstance(process_nofile_limit, int)
        or not 16 <= process_nofile_limit <= 1_048_576
    ):
        raise ValueError("process_nofile_limit must be an integer in [16,1048576]")
    if (
        isinstance(process_restart_limit, bool)
        or not isinstance(process_restart_limit, int)
        or not 1 <= process_restart_limit <= 10_000
    ):
        raise ValueError("process_restart_limit must be an integer in [1,10000]")
    if (
        isinstance(process_restart_window_s, bool)
        or not isinstance(process_restart_window_s, (int, float))
        or not math.isfinite(process_restart_window_s)
        or not 0 < process_restart_window_s <= 86_400
    ):
        raise ValueError("process_restart_window_s must be finite and in (0,86400]")
    if (
        isinstance(shutdown_timeout_s, bool)
        or not isinstance(shutdown_timeout_s, (int, float))
        or not math.isfinite(shutdown_timeout_s)
        or shutdown_timeout_s < 0
    ):
        raise ValueError("shutdown_timeout_s must be finite and >= 0")

    executor = BoundedInferenceExecutor(
        engine,
        capacity=queue_capacity,
        policy=OverflowPolicy(queue_policy),
        workers=inference_workers,
        execution_mode=resolved_execution_mode,
        execution_timeout_s=execution_timeout_s,
        process_control_timeout_s=process_control_timeout_s,
        process_memory_limit_mb=process_memory_limit_mb,
        process_nofile_limit=process_nofile_limit,
        process_restart_limit=process_restart_limit,
        process_restart_window_s=process_restart_window_s,
    )

    @asynccontextmanager
    async def lifespan(_app):
        executor.start()
        try:
            yield
        finally:
            executor.close(wait=True, timeout_s=shutdown_timeout_s)

    app = FastAPI(
        title="Edge AI Runtime",
        version="1.0.0",
        docs_url="/docs" if development else None,
        redoc_url="/redoc" if development else None,
        openapi_url="/openapi.json" if development else None,
        lifespan=lifespan,
    )
    app.state.inference_executor = executor
    # Starlette places the most recently added middleware outermost. Authenticate first,
    # then reserve a bounded prediction slot, then consume at most max_request_bytes.
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_request_bytes)
    app.add_middleware(
        RequestConcurrencyLimitMiddleware,
        max_inflight=queue_capacity + inference_workers,
    )
    app.add_middleware(APIKeyMiddleware, api_key=api_key or "", development=development)

    def auth(x_api_key: str | None):
        if development:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(401, "unauthorized")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(x_api_key: str | None = Header(None)):
        auth(x_api_key)
        try:
            executor_state = executor.ensure_ready()
        except EdgeAIError as e:
            raise HTTPException(503, f"inference executor unavailable: {e}") from e
        readiness = engine.readiness()
        readiness["inference_executor"] = executor_state
        if not readiness["ready"]:
            raise HTTPException(503, readiness)
        return readiness

    @app.get("/metrics")
    def metrics(x_api_key: str | None = Header(None)):
        auth(x_api_key)
        values = engine.metrics.snapshot()
        values.update({f"inference_queue_{k}": v for k, v in executor.stats().items()})
        return values

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, x_api_key: str | None = Header(None)):
        auth(x_api_key)
        try:
            pred, logits, version = await executor.submit_async(
                req.inputs, deadline_s=queue_deadline_s
            )
        except QueueFull as e:
            raise HTTPException(429, str(e), headers={"Retry-After": "1"}) from e
        except ExecutionDeadlineExceeded as e:
            raise HTTPException(504, str(e), headers={"X-EdgeAI-Timeout": "execution"}) from e
        except DeadlineExceeded as e:
            raise HTTPException(504, str(e), headers={"X-EdgeAI-Timeout": "queue"}) from e
        except ValidationError as e:
            raise HTTPException(422, str(e)) from e
        except ExecutorClosed as e:
            raise HTTPException(503, str(e)) from e
        except EdgeAIError as e:
            raise HTTPException(503, str(e)) from e
        return {
            "predictions": [int(x) for x in pred],
            "logits": [[float(value) for value in row] for row in logits],
            "version": version,
        }

    return app
