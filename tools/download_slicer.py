import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class SlicerPackage:
    version: str
    revision: str
    archive_name: str
    url: str
    sha512: str
    root_directory: str


DEFAULT_VERSION = "5.12.2"
PACKAGES = {
    "5.12.2": SlicerPackage(
        version="5.12.2",
        revision="34625",
        archive_name="Slicer-5.12.2-linux-amd64.tar.gz",
        url=(
            "https://slicer-packages.kitware.com/api/v1/item/"
            "6a57ba36ca34ceb380ea95fc/download"
        ),
        sha512=(
            "d7e8e724c9e5c64442da92475002a3b1f7052d1bc9104d181c027437d71a52f5"
            "fa3045cbca846ab5a50e5062d9144c30b63a0e630d8a89af69b4260c5f0374c9"
        ),
        root_directory="Slicer-5.12.2-linux-amd64",
    )
}


class SlicerInstallError(RuntimeError):
    pass


def file_sha512(path):
    digest = hashlib.sha512()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path, expected_sha512):
    actual = file_sha512(path)
    if actual != expected_sha512:
        raise SlicerInstallError(
            f"SHA-512 mismatch for {path}: expected {expected_sha512}, got {actual}"
        )


def download_archive(package, cache_directory):
    cache_directory = Path(cache_directory).resolve()
    cache_directory.mkdir(parents=True, exist_ok=True)
    archive_path = cache_directory / package.archive_name
    if archive_path.is_file():
        try:
            verify_archive(archive_path, package.sha512)
            print(f"Using cached {archive_path}", file=sys.stderr)
            return archive_path
        except SlicerInstallError:
            archive_path.unlink()

    request = Request(package.url, headers={"User-Agent": "samm-ci/1"})
    print(
        f"Downloading Slicer {package.version} revision {package.revision}",
        file=sys.stderr,
    )
    for attempt in range(1, 4):
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{package.archive_name}.",
            suffix=".part",
            dir=cache_directory,
        )
        os.close(temporary_fd)
        temporary_path = Path(temporary_name)
        try:
            with urlopen(request, timeout=120) as response, temporary_path.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            verify_archive(temporary_path, package.sha512)
            os.replace(temporary_path, archive_path)
            return archive_path
        except (SlicerInstallError, TimeoutError, URLError) as exc:
            if attempt == 3:
                raise SlicerInstallError(
                    f"could not download a verified archive after {attempt} attempts: {exc}"
                ) from exc
            print(f"Download attempt {attempt} failed; retrying: {exc}", file=sys.stderr)
        finally:
            temporary_path.unlink(missing_ok=True)
    raise AssertionError("download retry loop exited unexpectedly")


def _portable_safe_filter(member, destination):
    destination = Path(destination).resolve()
    member_path = (destination / member.name).resolve()
    if not member_path.is_relative_to(destination):
        raise SlicerInstallError(f"Archive member escapes destination: {member.name}")
    if member.isdev() or member.isfifo():
        raise SlicerInstallError(f"Unsupported archive member: {member.name}")
    if member.issym() or member.islnk():
        link_base = member_path.parent if member.issym() else destination
        link_target = (link_base / member.linkname).resolve()
        if not link_target.is_relative_to(destination):
            raise SlicerInstallError(f"Archive link escapes destination: {member.name}")
    return member


def extract_archive(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        data_filter = getattr(tarfile, "data_filter", None)
        archive.extractall(
            destination,
            filter=data_filter if data_filter is not None else _portable_safe_filter,
        )


def _marker_matches(install_root, package):
    marker_path = install_root / ".samm-slicer-package.json"
    executable = install_root / "Slicer"
    if not marker_path.is_file() or not executable.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker == {
        "version": package.version,
        "revision": package.revision,
        "sha512": package.sha512,
    }


def _remove_invalid_install(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def install_slicer(version, destination, cache_directory):
    try:
        package = PACKAGES[version]
    except KeyError:
        supported = ", ".join(sorted(PACKAGES))
        raise SlicerInstallError(
            f"Unsupported Slicer version {version!r}; supported versions: {supported}"
        ) from None

    destination = Path(destination).resolve()
    install_root = destination / package.root_directory
    executable = install_root / "Slicer"
    if _marker_matches(install_root, package):
        return executable

    archive_path = download_archive(package, cache_directory)
    destination.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".slicer-extract-", dir=destination))
    try:
        extract_archive(archive_path, staging_root)
        extracted_root = staging_root / package.root_directory
        extracted_executable = extracted_root / "Slicer"
        if not extracted_executable.is_file():
            raise SlicerInstallError(
                f"Archive did not contain {package.root_directory}/Slicer"
            )
        _remove_invalid_install(install_root)
        os.replace(extracted_root, install_root)
        marker = {
            "version": package.version,
            "revision": package.revision,
            "sha512": package.sha512,
        }
        (install_root / ".samm-slicer-package.json").write_text(
            json.dumps(marker, indent=2) + "\n",
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return executable


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download and verify a pinned 3D Slicer Linux package"
    )
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=sorted(PACKAGES))
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(".ci/slicer"),
        help="Directory where the verified Slicer package is extracted",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path(".ci/slicer-cache"),
        help="Directory where the verified archive is cached",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    try:
        executable = install_slicer(
            arguments.version,
            arguments.destination,
            arguments.cache_directory,
        )
    except (OSError, SlicerInstallError, tarfile.TarError) as exc:
        raise SystemExit(f"Slicer installation failed: {exc}") from None
    print(executable)


if __name__ == "__main__":
    main()
