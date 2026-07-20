from dataclasses import dataclass
from pathlib import Path
from shutil import which
import os


def pixi_program():
    program = os.environ.get("SAMM_PIXI") or which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")
    if not Path(program).exists():
        raise RuntimeError(f"pixi executable not found: {program}")
    return program


@dataclass(frozen=True)
class RuntimeDefinition:
    backend: str
    environment: str

    def command(self, host, port, model_dir, device):
        return [
            pixi_program(),
            "run",
            "-e",
            self.environment,
            "python",
            "-m",
            "samm_server.worker",
            "--host",
            host,
            "--port",
            str(port),
            "--model-dir",
            str(model_dir),
            "--device",
            device,
        ]


DEFAULT_RUNTIMES = (
    RuntimeDefinition("fastsam", "fastsam"),
    RuntimeDefinition("sam1", "sam1"),
    RuntimeDefinition("sam2", "sam2"),
    RuntimeDefinition("mobile_sam", "mobile-sam"),
    RuntimeDefinition("medsam", "medsam"),
    RuntimeDefinition("medsam_text", "medsam"),
    RuntimeDefinition("medsam2", "medsam2"),
    RuntimeDefinition("sam3", "sam3"),
    RuntimeDefinition("medical_sam3", "medical-sam3"),
)


class RuntimeRegistry:
    def __init__(self, definitions=DEFAULT_RUNTIMES):
        self.definitions = {definition.backend: definition for definition in definitions}

    def command(self, backend, host, port, model_dir, device):
        return self.definitions[backend].command(host, port, model_dir, device)
