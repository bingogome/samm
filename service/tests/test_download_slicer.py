import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.download_slicer import (
    PACKAGES,
    SlicerInstallError,
    SlicerPackage,
    extract_archive,
    file_sha512,
    install_slicer,
    verify_archive,
)


class DownloadSlicerTest(unittest.TestCase):
    def test_verifies_sha512(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "archive.tar.gz"
            archive.write_bytes(b"known archive contents")

            verify_archive(archive, file_sha512(archive))
            with self.assertRaisesRegex(SlicerInstallError, "SHA-512 mismatch"):
                verify_archive(archive, "0" * 128)

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("../escaped.txt")
                payload = b"must not escape"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            with self.assertRaises((SlicerInstallError, tarfile.FilterError)):
                extract_archive(archive_path, root / "destination")
            self.assertFalse((root / "escaped.txt").exists())

    def test_installs_and_reuses_verified_local_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_archive = root / "Slicer-test-linux-amd64.tar.gz"
            executable_payload = b"#!/bin/sh\nexit 0\n"
            with tarfile.open(source_archive, "w:gz") as archive:
                executable = tarfile.TarInfo("Slicer-test-linux-amd64/Slicer")
                executable.mode = 0o755
                executable.size = len(executable_payload)
                archive.addfile(executable, io.BytesIO(executable_payload))
            package = SlicerPackage(
                version="test",
                revision="1",
                archive_name=source_archive.name,
                url=source_archive.as_uri(),
                sha512=file_sha512(source_archive),
                root_directory="Slicer-test-linux-amd64",
            )
            destination = root / "installed"
            cache = root / "cache"

            with patch.dict(PACKAGES, {"test": package}):
                first = install_slicer("test", destination, cache)
                source_archive.unlink()
                second = install_slicer("test", destination, cache)

            self.assertEqual(first, second)
            self.assertEqual(first.read_bytes(), executable_payload)
            self.assertTrue(first.stat().st_mode & 0o100)
            self.assertTrue((first.parent / ".samm-slicer-package.json").is_file())


if __name__ == "__main__":
    unittest.main()
