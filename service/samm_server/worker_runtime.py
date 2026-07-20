import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import HEALTH_PATH
from .runtime_registry import RuntimeRegistry


@dataclass(frozen=True)
class ProxyResult:
    status_code: int
    payload: dict


class WorkerRuntime:
    def __init__(self, host, port, model_dir, device, registry=None, startup_timeout=30):
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.device = device
        self.registry = registry or RuntimeRegistry()
        self.startup_timeout = startup_timeout
        self.backend = None
        self.command = None
        self.process = None

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def use_backend(self, backend):
        command = self.registry.command(backend, self.host, self.port, self.model_dir, self.device)
        if self.backend == backend and self.healthy():
            return
        self.stop()
        self.backend = backend
        self.command = command
        self.start()

    def ensure_running(self):
        if not self.command:
            raise RuntimeError("worker backend is not selected")
        if self.healthy():
            return
        self.start()

    def start(self):
        print(f"SAMM worker command: {shlex.join(self.command)}", flush=True)
        self.process = subprocess.Popen(self.command)
        self.wait_until_ready()

    def healthy(self):
        try:
            with urlopen(f"{self.url}{HEALTH_PATH}", timeout=1) as response:
                return response.status == 200
        except URLError:
            return False

    def wait_until_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"worker exited: {' '.join(self.command)}")
            if self.healthy():
                return
            time.sleep(0.1)
        raise TimeoutError(f"worker did not start: {' '.join(self.command)}")

    def stop(self):
        if self.process and self.process.poll() is None:
            print(f"SAMM worker stopping pid={self.process.pid}", flush=True)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"SAMM worker killing pid={self.process.pid}", flush=True)
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def offload(self):
        self.stop()
        self.backend = None
        self.command = None


class WorkerProxy:
    def __init__(self, runtime, timeout=600):
        self.runtime = runtime
        self.timeout = timeout

    def request(self, method, path, payload=None):
        self.runtime.ensure_running()
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(f"{self.runtime.url}{path}", data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return ProxyResult(response.status, json.loads(response.read().decode("utf-8")))
        except HTTPError as exc:
            return ProxyResult(exc.code, json.loads(exc.read().decode("utf-8")))
        except URLError as exc:
            raise ConnectionError(f"worker unavailable: {exc.reason}") from None
