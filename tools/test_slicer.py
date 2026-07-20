import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def discover_slicer(root, explicit=None, environment=None):
    root = Path(root).resolve()
    environment = os.environ if environment is None else environment
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if environment.get("SLICER_EXECUTABLE"):
        candidates.append(Path(environment["SLICER_EXECUTABLE"]).expanduser())
    candidates.extend((root.parent / "Slicer" / "Slicer", root / "Slicer" / "Slicer"))
    for command in ("Slicer", "slicer"):
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Slicer executable was not found. Set SLICER_EXECUTABLE or pass --slicer /path/to/Slicer."
    )


def slicer_command(executable, root):
    root = Path(root).resolve()
    module_path = root / "samm" / "SegmentAnyMedicalModel"
    test_script = root / "tools" / "slicer_integration_tests.py"
    return [
        str(executable),
        "--no-splash",
        "--no-main-window",
        "--disable-cli-modules",
        "--disable-settings",
        "--ignore-slicerrc",
        "--additional-module-path",
        str(module_path),
        "--python-script",
        str(test_script),
    ]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SAMM integration tests inside Slicer's Python environment"
    )
    parser.add_argument("--slicer", help="Path to the Slicer launcher executable")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum runtime in seconds")
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        executable = discover_slicer(root, arguments.slicer)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    command = slicer_command(executable, root)
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["SAMM_TEST_ROOT"] = str(root)
    print(f"Running: {shlex.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            timeout=arguments.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"Slicer integration tests timed out after {arguments.timeout} seconds"
        ) from exc
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
