"""Host-side supervisor for out-of-process Zaxy plugins.

Each out-of-process plugin gets one long-lived ``python -m zaxy.plugin_worker``
child. The host talks to it over newline-delimited JSON (:mod:`zaxy.plugin_ipc`)
and never imports the plugin module itself.

What this buys, precisely:

* **Fault isolation.** A plugin that raises at import, or segfaults mid-call,
  kills only its own process. The host observes a closed pipe, marks the worker
  dead, records a degraded operation, and keeps serving.
* **Liveness.** Every request is bounded by a deadline. A worker that stops
  answering is killed and marked dead, so one hung plugin cannot wedge ingestion.

What this does **not** buy: it is not a security sandbox. The child runs as the
same user with the same filesystem and network access as the host, and it is
handed event payloads. Untrusted plugin code needs an OS-level sandbox
(containers, seccomp, a separate user) on top of this.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zaxy.log import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.config import Settings
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult
    from zaxy.plugins import PluginAPI, PluginLoadResult

logger = get_logger("plugins.subprocess")

_EOF = object()


class PluginWorkerError(RuntimeError):
    """Base class for out-of-process plugin transport failures."""


class PluginWorkerCrashedError(PluginWorkerError):
    """The worker process exited or closed its pipe unexpectedly."""


class PluginWorkerTimeoutError(PluginWorkerError):
    """The worker did not answer within its deadline and was killed."""


@dataclass(frozen=True)
class RemotePluginDescription:
    """What a worker reports about its plugin during the handshake."""

    name: str
    version: str
    protocol: int
    event_types: tuple[str, ...]
    unsupported_backends: tuple[str, ...] = ()


class PluginWorker:
    """A supervised child process serving exactly one plugin.

    The worker is spawned lazily on the first request. Once it dies — crash,
    timeout, or explicit close — it stays dead: :attr:`dead_reason` is set and
    every later request raises :class:`PluginWorkerCrashedError` immediately rather
    than respawning code that has already proven unhealthy.
    """

    def __init__(self, reference: str, *, timeout: float) -> None:
        self.reference = reference
        self.timeout = timeout
        self.dead_reason: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()

    def _spawn(self) -> subprocess.Popen[str]:
        """Start the worker process and its stdout reader thread."""
        process = subprocess.Popen(
            [sys.executable, "-m", "zaxy.plugin_worker", self.reference],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        def _pump(stream: Any, sink: queue.Queue[Any]) -> None:
            try:
                for line in stream:
                    sink.put(line)
            finally:
                sink.put(_EOF)

        thread = threading.Thread(
            target=_pump,
            args=(process.stdout, self._lines),
            daemon=True,
            name=f"zaxy-plugin-{self.reference}",
        )
        thread.start()
        self._process = process
        return process

    def _kill(self, reason: str) -> None:
        """Terminate the worker and mark it permanently dead."""
        self.dead_reason = reason
        process = self._process
        if process is None:
            return
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:  # pragma: no cover - best-effort teardown
            logger.warning("Failed to reap plugin worker %r", self.reference)
        self._process = None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return the decoded response.

        Raises:
            PluginWorkerCrashedError: the worker is dead or its pipe closed.
            PluginWorkerTimeoutError: no response arrived within ``timeout``.
        """
        with self._lock:
            if self.dead_reason is not None:
                raise PluginWorkerCrashedError(self.dead_reason)
            process = self._process or self._spawn()

            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                self._kill(f"worker pipe closed while sending: {exc}")
                raise PluginWorkerCrashedError(str(self.dead_reason)) from exc

            try:
                line = self._lines.get(timeout=self.timeout)
            except queue.Empty:
                self._kill(f"worker exceeded {self.timeout}s deadline")
                raise PluginWorkerTimeoutError(str(self.dead_reason)) from None

            if line is _EOF:
                code = process.poll()
                self._kill(f"worker exited (returncode={code}) without responding")
                raise PluginWorkerCrashedError(str(self.dead_reason))

            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                self._kill(f"worker emitted malformed response: {exc}")
                raise PluginWorkerCrashedError(str(self.dead_reason)) from exc

            if not isinstance(decoded, dict):
                self._kill("worker response was not a JSON object")
                raise PluginWorkerCrashedError(str(self.dead_reason))
            return decoded

    def describe(self) -> RemotePluginDescription:
        """Handshake with the worker and return what its plugin provides."""
        response = self.request({"op": "describe"})
        if not response.get("ok"):
            raise PluginWorkerError(str(response.get("error") or "describe failed"))
        return RemotePluginDescription(
            name=str(response.get("name") or self.reference),
            version=str(response.get("version") or ""),
            protocol=int(response.get("protocol") or 0),
            event_types=tuple(str(item) for item in response.get("event_types") or ()),
            unsupported_backends=tuple(
                str(item) for item in response.get("unsupported_backends") or ()
            ),
        )

    def extract(self, event_type: str, event: Event) -> ExtractionResult:
        """Run one remote extraction and decode the result."""
        from zaxy.plugin_ipc import decode_extraction_result, encode_event

        response = self.request(
            {"op": "extract", "event_type": event_type, "event": encode_event(event)}
        )
        if not response.get("ok"):
            raise PluginWorkerError(str(response.get("error") or "extract failed"))
        return decode_extraction_result(response.get("result"))

    def close(self) -> None:
        """Shut the worker down cooperatively, then forcibly."""
        with self._lock:
            process = self._process
            if process is None:
                self.dead_reason = self.dead_reason or "closed"
                return
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=self.timeout)
                self._process = None
                self.dead_reason = self.dead_reason or "closed"
            except Exception:
                self._kill("closed forcibly")


# Workers created by load_out_of_process_plugins, keyed by reference, so repeated
# loads reuse one child per plugin and tests can reach in and shut them down.
_WORKERS: dict[str, PluginWorker] = {}


def active_workers() -> dict[str, PluginWorker]:
    """Return the live worker registry keyed by plugin reference."""
    return _WORKERS


def shutdown_workers() -> None:
    """Close every supervised worker and clear the registry."""
    for worker in list(_WORKERS.values()):
        worker.close()
    _WORKERS.clear()


def _degrade(reason: str) -> None:
    """Record a degraded plugin operation without letting metrics failures escape."""
    try:
        from zaxy.metrics import get_metrics

        get_metrics().record_degraded_operation("plugin_out_of_process", reason)
    except Exception:  # pragma: no cover - metrics must never break ingestion
        logger.debug("Unable to record degraded plugin operation %r", reason)


def _remote_extractor(
    worker: PluginWorker,
    event_type: str,
) -> Callable[[Event], ExtractionResult]:
    """Build a host-side stub that forwards ``event_type`` to ``worker``.

    A transport failure degrades to an empty extraction for that event rather
    than propagating: one sick plugin must not fail the whole ingest.
    """

    def _extract(event: Event) -> ExtractionResult:
        from zaxy.extract import ExtractionResult

        try:
            return worker.extract(event_type, event)
        except PluginWorkerError as exc:
            logger.warning(
                "Out-of-process plugin %r failed on %s: %s", worker.reference, event_type, exc
            )
            _degrade(f"{worker.reference}:{type(exc).__name__}")
            return ExtractionResult(entities=[], edges=[], source_event_seq=event.seq)

    return _extract


def load_out_of_process_plugins(
    settings: Settings,
    api: PluginAPI,
) -> list[PluginLoadResult]:
    """Start a worker per ``settings.plugins_out_of_process`` entry and register stubs.

    Registration installs *host-side stubs*; no plugin module is imported here.
    A worker that fails its handshake is reported as failed and its plugin is
    simply absent — never fatal.
    """
    from zaxy.plugins import PluginLoadResult

    results: list[PluginLoadResult] = []
    timeout = float(settings.plugin_timeout_seconds)

    for raw in settings.plugins_out_of_process:
        reference = raw.strip()
        if not reference:
            continue
        worker = _WORKERS.get(reference)
        if worker is not None and worker.dead_reason is None:
            results.append(
                PluginLoadResult(
                    name=reference, version="", source="subprocess", status="loaded"
                )
            )
            continue

        worker = PluginWorker(reference, timeout=timeout)
        try:
            description = worker.describe()
        except (PluginWorkerError, OSError) as exc:
            worker.close()
            logger.warning("Out-of-process plugin %r failed to start: %s", reference, exc)
            _degrade(f"{reference}:load_failed")
            results.append(
                PluginLoadResult(
                    name=reference,
                    version="",
                    source="subprocess",
                    status="failed",
                    error=str(exc),
                )
            )
            continue

        _WORKERS[reference] = worker
        for event_type in description.event_types:
            api.register_extractor(event_type, _remote_extractor(worker, event_type))
        if description.unsupported_backends:
            logger.warning(
                "Out-of-process plugin %r requested projection backends %s, which cannot "
                "cross a process boundary and were not registered",
                reference,
                ", ".join(description.unsupported_backends),
            )
        results.append(
            PluginLoadResult(
                name=description.name,
                version=description.version,
                source="subprocess",
                status="loaded",
            )
        )

    return results
