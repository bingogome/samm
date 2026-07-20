from pathlib import Path
import tempfile
import unittest

from tools.test_slicer import discover_slicer, slicer_command


class SlicerLauncherTest(unittest.TestCase):
    def test_discovers_explicit_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "CustomSlicer"
            executable.touch()
            executable.chmod(0o755)

            self.assertEqual(discover_slicer(root, executable, {}), executable.resolve())

    def test_discovers_environment_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "EnvironmentSlicer"
            executable.touch()
            executable.chmod(0o755)

            result = discover_slicer(
                root, environment={"SLICER_EXECUTABLE": str(executable)}
            )

            self.assertEqual(result, executable.resolve())

    def test_command_is_isolated_and_uses_source_module(self):
        root = Path("/samm").resolve()
        command = slicer_command(Path("/opt/Slicer/Slicer"), root)

        self.assertIn("--disable-settings", command)
        self.assertIn("--ignore-slicerrc", command)
        self.assertEqual(
            command[command.index("--additional-module-path") + 1],
            str(root / "samm" / "SegmentAnyMedicalModel"),
        )
        self.assertEqual(
            command[command.index("--python-script") + 1],
            str(root / "tools" / "slicer_integration_tests.py"),
        )


if __name__ == "__main__":
    unittest.main()
