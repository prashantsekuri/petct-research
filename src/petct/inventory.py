"""Read-only inventory of DICOM studies and series."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pydicom
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue


METADATA_FIELDS = (
    "StudyDescription",
    "SeriesDescription",
    "Modality",
    "SOPClassUID",
    "Manufacturer",
    "ManufacturerModelName",
    "Rows",
    "Columns",
    "SliceThickness",
    "PixelSpacing",
)
GEOMETRY_FIELDS = ("Rows", "Columns", "SliceThickness", "PixelSpacing")
READ_TAGS = ("StudyInstanceUID", "SeriesInstanceUID", *METADATA_FIELDS)


def _value(dataset: Dataset, keyword: str) -> Any:
    return dataset.get(keyword)


def _display_value(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, MultiValue):
        return " x ".join(str(item) for item in value)
    return str(value)


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, MultiValue):
        left = tuple(left)
    if isinstance(right, MultiValue):
        right = tuple(right)
    return left == right


@dataclass
class SeriesInventory:
    """Allowlisted metadata accumulated for one DICOM series."""

    study_instance_uid: str
    series_instance_uid: str
    metadata: dict[str, Any] = field(default_factory=dict)
    number_of_instances: int = 0
    geometry_values: dict[str, list[Any]] = field(
        default_factory=lambda: {name: [] for name in GEOMETRY_FIELDS}
    )

    def add(self, dataset: Dataset) -> None:
        self.number_of_instances += 1

        for name in METADATA_FIELDS:
            value = _value(dataset, name)
            if name not in self.metadata or self.metadata[name] in (None, ""):
                self.metadata[name] = value

        for name in GEOMETRY_FIELDS:
            value = _value(dataset, name)
            observed = self.geometry_values[name]
            if not any(_values_match(value, existing) for existing in observed):
                observed.append(value)

    @property
    def varying_geometry_fields(self) -> list[str]:
        return [
            name for name in GEOMETRY_FIELDS if len(self.geometry_values[name]) > 1
        ]


@dataclass
class InventoryResult:
    """Grouped inventory plus aggregate scan counts."""

    studies: dict[str, dict[str, SeriesInventory]] = field(default_factory=dict)
    scanned_files: int = 0
    dicom_objects: int = 0
    skipped_files: int = 0
    unreadable_directories: int = 0


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _validate_source(source: Path) -> Path:
    try:
        resolved = source.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError("source must be an existing, accessible directory") from error

    if not resolved.is_dir():
        raise ValueError("source must be a directory")

    project_root = _project_root()
    if resolved == project_root or project_root in resolved.parents:
        raise ValueError("source patient data must be outside the repository")

    return resolved


def inventory_directory(source: Path) -> InventoryResult:
    """Read allowlisted metadata from DICOM files beneath an external directory."""

    source = _validate_source(source)
    result = InventoryResult()

    def note_walk_error(_error: OSError) -> None:
        result.unreadable_directories += 1

    for root, directories, filenames in os.walk(
        source, topdown=True, onerror=note_walk_error, followlinks=False
    ):
        directories.sort()
        filenames.sort()

        for filename in filenames:
            path = Path(root, filename)
            if path.is_symlink():
                result.skipped_files += 1
                continue

            result.scanned_files += 1
            try:
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=READ_TAGS,
                    force=False,
                )
            except Exception:
                result.skipped_files += 1
                continue

            study_uid = _value(dataset, "StudyInstanceUID")
            series_uid = _value(dataset, "SeriesInstanceUID")
            if not study_uid or not series_uid:
                result.skipped_files += 1
                continue

            study_key = str(study_uid)
            series_key = str(series_uid)
            series_by_uid = result.studies.setdefault(study_key, {})
            series = series_by_uid.setdefault(
                series_key,
                SeriesInventory(
                    study_instance_uid=study_key,
                    series_instance_uid=series_key,
                ),
            )
            series.add(dataset)
            result.dicom_objects += 1

    return result


def format_inventory(result: InventoryResult) -> str:
    """Format an inventory without exposing non-allowlisted metadata."""

    lines: list[str] = []
    for study_uid in sorted(result.studies):
        series_by_uid = result.studies[study_uid]
        first_series = next(iter(series_by_uid.values()))
        lines.extend(
            [
                f"Study: {study_uid}",
                "  StudyDescription: "
                f"{_display_value(first_series.metadata.get('StudyDescription'))}",
                f"  NumberOfSeries: {len(series_by_uid)}",
            ]
        )

        for series_uid in sorted(series_by_uid):
            series = series_by_uid[series_uid]
            lines.append(f"  Series: {series_uid}")
            for name in METADATA_FIELDS:
                if name == "StudyDescription":
                    continue
                lines.append(
                    f"    {name}: {_display_value(series.metadata.get(name))}"
                )
            lines.append(f"    NumberOfInstances: {series.number_of_instances}")

            varying = series.varying_geometry_fields
            status = "WARNING" if varying else "OK"
            lines.append(f"    GeometryConsistency: {status}")
            if varying:
                lines.append(f"    VaryingGeometryFields: {', '.join(varying)}")

    if not result.studies:
        lines.append("No strictly parsed DICOM studies found.")

    lines.extend(
        [
            "",
            "Scan summary:",
            f"  FilesScanned: {result.scanned_files}",
            f"  DICOMObjects: {result.dicom_objects}",
            f"  SkippedFiles: {result.skipped_files}",
            f"  UnreadableDirectories: {result.unreadable_directories}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory DICOM studies and series without reading pixel data."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="source DICOM directory located outside this repository",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = inventory_directory(args.source)
    except ValueError as error:
        parser.error(str(error))
    print(format_inventory(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
