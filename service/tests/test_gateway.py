import unittest
from pathlib import Path
from unittest.mock import patch

from samm_server.runtime_registry import RuntimeDefinition, RuntimeRegistry


class GatewayTest(unittest.TestCase):
    def test_runtime_definition_command(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            self.assertEqual(
                RuntimeDefinition("sam1", "sam1").command("127.0.0.1", 8801, Path("checkpoints"), "cuda"),
                [
                    "/opt/pixi",
                    "run",
                    "-e",
                    "sam1",
                    "python",
                    "-m",
                    "samm_server.worker",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8801",
                    "--model-dir",
                    "checkpoints",
                    "--device",
                    "cuda",
                ],
            )

    def test_runtime_registry_uses_backend_definition(self):
        registry = RuntimeRegistry((RuntimeDefinition("mobile_sam", "mobile-sam"),))

        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            self.assertEqual(
                registry.command("mobile_sam", "127.0.0.1", 8802, "models", "cpu"),
                [
                    "/opt/pixi",
                    "run",
                    "-e",
                    "mobile-sam",
                    "python",
                    "-m",
                    "samm_server.worker",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8802",
                    "--model-dir",
                    "models",
                    "--device",
                    "cpu",
                ],
            )

    def test_default_runtime_registry_supports_mobile_sam(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("mobile_sam", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "mobile-sam")

    def test_default_runtime_registry_supports_sam2(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("sam2", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "sam2")

    def test_default_runtime_registry_supports_medsam(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("medsam", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "medsam")

    def test_default_runtime_registry_supports_medsam_text(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("medsam_text", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "medsam")

    def test_default_runtime_registry_supports_medsam2(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("medsam2", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "medsam2")

    def test_default_runtime_registry_supports_sam3(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("sam3", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "sam3")

    def test_default_runtime_registry_supports_medical_sam3(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("medical_sam3", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "medical-sam3")

    def test_default_runtime_registry_supports_fastsam(self):
        with patch("samm_server.runtime_registry.pixi_program", return_value="/opt/pixi"):
            command = RuntimeRegistry().command("fastsam", "127.0.0.1", 8802, "models", "cpu")

        self.assertEqual(command[3], "fastsam")

    def test_runtime_registry_rejects_unknown_backend(self):
        with self.assertRaises(KeyError):
            RuntimeRegistry().command("unknown", "127.0.0.1", 8801, "models", "cpu")


if __name__ == "__main__":
    unittest.main()
