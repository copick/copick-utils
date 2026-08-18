"""Fail release builds whose artifacts do not expose the migrated contract."""

import argparse
import email
import tarfile
import zipfile
from pathlib import Path

EXPECTED_REQUIREMENTS = {
    "copick>=2.0.0a1",
    "pydantic>=2",
    "zarr<4,>=3.1.6",
}
EXPECTED_ENTRY_POINTS = 32


def inspect_distributions(dist_dir: Path) -> tuple[Path, Path]:
    wheels = list(dist_dir.glob("*.whl"))
    sdists = list(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"Expected one wheel and one source distribution, found {wheels!r} and {sdists!r}")

    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        entry_point_names = [name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")]
        if len(metadata_names) != 1 or len(entry_point_names) != 1:
            raise ValueError("Wheel must contain exactly one METADATA and one entry_points.txt file")

        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        requirements = {value.replace(" ", "") for value in metadata.get_all("Requires-Dist", [])}
        missing = EXPECTED_REQUIREMENTS - requirements
        if missing:
            raise ValueError(f"Wheel is missing migration requirements: {sorted(missing)!r}")
        if metadata["Requires-Python"] != ">=3.11":
            raise ValueError(f"Unexpected Requires-Python: {metadata['Requires-Python']!r}")

        entry_points = archive.read(entry_point_names[0]).decode()
        command_count = sum(
            1 for line in entry_points.splitlines() if line and not line.startswith("[") and "=" in line
        )
        if command_count != EXPECTED_ENTRY_POINTS:
            raise ValueError(f"Expected {EXPECTED_ENTRY_POINTS} command entry points, found {command_count}")

    sdist = sdists[0]
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        if not any(name.endswith("/uv.lock") for name in names):
            raise ValueError("Source distribution does not contain uv.lock")

    return wheel, sdist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    args = parser.parse_args()
    wheel, sdist = inspect_distributions(args.dist_dir)
    print(f"Validated {wheel.name} and {sdist.name}")


if __name__ == "__main__":
    main()
