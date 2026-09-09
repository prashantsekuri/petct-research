"""Read-only inventory of DICOM studies and series."""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
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
PET_OBSERVATION_FIELDS = (
    "Units",
    "SUVType",
    "RescaleSlope",
    "RescaleIntercept",
    "CorrectedImage",
    "DecayCorrection",
    "PatientWeight",
    "PatientSize",
    "PatientSex",
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "TimezoneOffsetFromUTC",
    "RadiopharmaceuticalStartTime",
    "RadiopharmaceuticalStartDateTime",
)
TIMING_FIELDS = (
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "TimezoneOffsetFromUTC",
    "RadiopharmaceuticalStartTime",
    "RadiopharmaceuticalStartDateTime",
)
RADIOPHARMACEUTICAL_FIELDS = (
    "Radiopharmaceutical",
    "RadionuclideTotalDose",
    "RadionuclideHalfLife",
    "RadiopharmaceuticalStartTime",
    "RadiopharmaceuticalStartDateTime",
)
RWV_SEQUENCE_FIELDS = (
    "ReferencedImageRealWorldValueMappingSequence",
    "RealWorldValueMappingSequence",
)
RWV_MAPPING_FIELDS = (
    "RealWorldValueFirstValueMapped",
    "RealWorldValueLastValueMapped",
    "RealWorldValueSlope",
    "RealWorldValueIntercept",
)
PIXEL_PROBE_FIELDS = (
    "InstanceNumber",
    "ImagePositionPatient",
    "SliceLocation",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
)
CODE_FIELDS = (
    "CodeValue",
    "LongCodeValue",
    "URNCodeValue",
    "CodingSchemeDesignator",
    "CodingSchemeVersion",
    "CodeMeaning",
)
PET_READ_TAGS = tuple(
    dict.fromkeys(
        (
            *READ_TAGS,
            "SOPInstanceUID",
            *PET_OBSERVATION_FIELDS,
            *PIXEL_PROBE_FIELDS,
            "RadiopharmaceuticalInformationSequence",
            *RADIOPHARMACEUTICAL_FIELDS,
            *RWV_SEQUENCE_FIELDS,
            *RWV_MAPPING_FIELDS,
            "MeasurementUnitsCodeSequence",
            "QuantityDefinitionSequence",
            "ConceptNameCodeSequence",
            "ConceptCodeSequence",
            *CODE_FIELDS,
            "ReferencedSeriesSequence",
            "ReferencedImageSequence",
            "ReferencedInstanceSequence",
            "ReferencedSOPInstanceUID",
            "ReferencedSOPClassUID",
        )
    )
)


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


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or (
        isinstance(value, (MultiValue, list, tuple)) and len(value) == 0
    )


def _normalized_value(value: Any) -> str | tuple[str, ...] | None:
    if _is_empty(value):
        return None
    if isinstance(value, (MultiValue, list, tuple)):
        return tuple(str(item) for item in value)
    return str(value)


def _format_normalized(value: str | tuple[str, ...]) -> str:
    if isinstance(value, tuple):
        return "\\".join(value)
    return value


def _string_value(dataset: Dataset, keyword: str) -> str | None:
    value = _normalized_value(_value(dataset, keyword))
    return value if isinstance(value, str) else None


def _transfer_syntax_uid(dataset: Dataset) -> str | None:
    file_meta = getattr(dataset, "file_meta", None)
    if file_meta is None:
        return None
    return _string_value(file_meta, "TransferSyntaxUID")


@dataclass
class FieldObservation:
    """Distinct non-empty values and instance-level presence for one field."""

    values: list[str | tuple[str, ...]] = field(default_factory=list)
    present_instances: int = 0
    missing_instances: int = 0

    def add(self, value: Any) -> None:
        self.add_many([] if _is_empty(value) else [value])

    def add_many(self, values: Sequence[Any]) -> None:
        normalized = [
            item
            for value in values
            if (item := _normalized_value(value)) is not None
        ]
        if not normalized:
            self.missing_instances += 1
            return

        self.present_instances += 1
        for value in normalized:
            if value not in self.values:
                self.values.append(value)

    @property
    def state(self) -> str:
        if not self.values:
            return "MISSING"
        if len(self.values) > 1:
            return "VARYING"
        if self.missing_instances:
            return "PARTIAL"
        return "CONSTANT"

    @property
    def first_value(self) -> str | tuple[str, ...] | None:
        return self.values[0] if self.values else None


def _sequence_items(dataset: Dataset, keyword: str) -> list[Dataset]:
    value = _value(dataset, keyword)
    if not value:
        return []
    return [item for item in value if isinstance(item, Dataset)]


def _code_summary(dataset: Dataset) -> str:
    parts = []
    for name in CODE_FIELDS:
        value = _normalized_value(_value(dataset, name))
        if value is not None:
            parts.append(f"{name}={_format_normalized(value)}")
    return ", ".join(parts) if parts else "No allowlisted coded values"


def _mapping_suv_type(dataset: Dataset) -> str:
    code_values = []
    code_meanings = []
    code_sequences = _sequence_items(dataset, "MeasurementUnitsCodeSequence")
    for quantity in _sequence_items(dataset, "QuantityDefinitionSequence"):
        code_sequences.extend(_sequence_items(quantity, "ConceptCodeSequence"))
    for code in code_sequences:
        code_value = _string_value(code, "CodeValue")
        code_meaning = _string_value(code, "CodeMeaning")
        if code_value:
            code_values.append(code_value.upper())
        if code_meaning:
            code_meanings.append(code_meaning.upper())

    joined_values = " ".join(code_values)
    joined_meanings = " ".join(code_meanings)
    if "SUVLBM" in joined_values or "LEAN BODY MASS" in joined_meanings:
        return "SUVlbm"
    if "SUVBSA" in joined_values or "BODY SURFACE AREA" in joined_meanings:
        return "SUVbsa"
    if "SUVIBW" in joined_values or "IDEAL BODY WEIGHT" in joined_meanings:
        return "SUVibw"
    if "SUVBW" in joined_values or "BODY WEIGHT" in joined_meanings:
        return "SUVbw"
    if "STANDARDIZED UPTAKE VALUE" in joined_meanings:
        return "SUV (unspecified normalization)"
    return "Unknown"


@dataclass
class RadiopharmaceuticalItem:
    values: dict[str, str | tuple[str, ...] | None]

    @property
    def signature(self) -> tuple[Any, ...]:
        return tuple(self.values[name] for name in RADIOPHARMACEUTICAL_FIELDS)


@dataclass(frozen=True)
class PTInstanceRecord:
    """Metadata-only handle for a possible bounded pixel probe."""

    path: Path = field(repr=False)
    sop_instance_uid: str
    instance_number: str | None
    image_position_patient: tuple[str, ...] | None
    slice_location: str | None
    acquisition_time: str | None
    rescale_slope: str | None
    rescale_intercept: str | None
    bits_allocated: str | None
    bits_stored: str | None
    high_bit: str | None
    pixel_representation: str | None
    transfer_syntax_uid: str | None

    @classmethod
    def from_dataset(cls, dataset: Dataset, path: Path) -> PTInstanceRecord | None:
        sop_instance_uid = _normalized_value(_value(dataset, "SOPInstanceUID"))
        if not isinstance(sop_instance_uid, str):
            return None

        position = _normalized_value(_value(dataset, "ImagePositionPatient"))
        return cls(
            path=path,
            sop_instance_uid=sop_instance_uid,
            instance_number=_string_value(dataset, "InstanceNumber"),
            image_position_patient=(
                position if isinstance(position, tuple) else None
            ),
            slice_location=_string_value(dataset, "SliceLocation"),
            acquisition_time=_string_value(dataset, "AcquisitionTime"),
            rescale_slope=_string_value(dataset, "RescaleSlope"),
            rescale_intercept=_string_value(dataset, "RescaleIntercept"),
            bits_allocated=_string_value(dataset, "BitsAllocated"),
            bits_stored=_string_value(dataset, "BitsStored"),
            high_bit=_string_value(dataset, "HighBit"),
            pixel_representation=_string_value(dataset, "PixelRepresentation"),
            transfer_syntax_uid=_transfer_syntax_uid(dataset),
        )


@dataclass
class PTDetails:
    """Per-instance observations needed for later PET quantitation validation."""

    observations: dict[str, FieldObservation] = field(
        default_factory=lambda: {
            name: FieldObservation() for name in PET_OBSERVATION_FIELDS
        }
    )
    radiopharmaceutical_sequence: FieldObservation = field(
        default_factory=FieldObservation
    )
    radiopharmaceutical_observations: dict[str, FieldObservation] = field(
        default_factory=lambda: {
            name: FieldObservation() for name in RADIOPHARMACEUTICAL_FIELDS
        }
    )
    radiopharmaceutical_items: list[RadiopharmaceuticalItem] = field(
        default_factory=list
    )
    instances: list[PTInstanceRecord] = field(default_factory=list)

    def add(
        self, dataset: Dataset, *, path: Path | None = None, retain_instance: bool = False
    ) -> None:
        for name, observation in self.observations.items():
            observation.add(_value(dataset, name))

        sequence_items = _sequence_items(
            dataset, "RadiopharmaceuticalInformationSequence"
        )
        self.radiopharmaceutical_sequence.add(
            "present" if sequence_items else None
        )
        for name, observation in self.radiopharmaceutical_observations.items():
            observation.add_many([_value(item, name) for item in sequence_items])

        for item in sequence_items:
            radiopharmaceutical = RadiopharmaceuticalItem(
                values={
                    name: _normalized_value(_value(item, name))
                    for name in RADIOPHARMACEUTICAL_FIELDS
                }
            )
            if all(
                radiopharmaceutical.signature != existing.signature
                for existing in self.radiopharmaceutical_items
            ):
                self.radiopharmaceutical_items.append(radiopharmaceutical)

        if retain_instance and path is not None:
            modality = _string_value(dataset, "Modality")
            units = _string_value(dataset, "Units")
            if modality == "PT" and units == "BQML":
                record = PTInstanceRecord.from_dataset(dataset, path)
                if record is not None:
                    self.instances.append(record)


@dataclass
class RWVMapping:
    suv_type: str
    first_value_mapped: str | tuple[str, ...] | None
    last_value_mapped: str | tuple[str, ...] | None
    slope: str | tuple[str, ...] | None
    intercept: str | tuple[str, ...] | None
    measurement_units: tuple[str, ...]
    quantity_definitions: tuple[str, ...]

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.suv_type,
            self.first_value_mapped,
            self.last_value_mapped,
            self.slope,
            self.intercept,
            self.measurement_units,
            self.quantity_definitions,
        )


@dataclass
class RWVDetails:
    """Allowlisted real-world-value mappings and source references."""

    sequence_presence: dict[str, FieldObservation] = field(
        default_factory=lambda: {
            name: FieldObservation() for name in RWV_SEQUENCE_FIELDS
        }
    )
    mappings: list[RWVMapping] = field(default_factory=list)
    referenced_series_uids: list[str] = field(default_factory=list)
    referenced_sop_instance_uids: list[str] = field(default_factory=list)

    def add(self, dataset: Dataset) -> None:
        for name, observation in self.sequence_presence.items():
            items = _sequence_items(dataset, name)
            observation.add("present" if items else None)
            self._collect_mappings(items)
            self._collect_references(items)

        referenced_series = _sequence_items(dataset, "ReferencedSeriesSequence")
        self._collect_references(referenced_series)
        self._collect_references(_sequence_items(dataset, "ReferencedImageSequence"))
        self._collect_references(
            _sequence_items(dataset, "ReferencedInstanceSequence")
        )

    def _collect_mappings(self, items: Sequence[Dataset]) -> None:
        pending = list(items)
        while pending:
            item = pending.pop(0)
            nested = _sequence_items(item, "RealWorldValueMappingSequence")
            pending.extend(nested)

            has_mapping = any(_value(item, name) is not None for name in RWV_MAPPING_FIELDS)
            has_mapping = has_mapping or bool(
                _sequence_items(item, "MeasurementUnitsCodeSequence")
                or _sequence_items(item, "QuantityDefinitionSequence")
            )
            if not has_mapping:
                continue

            units = tuple(
                _code_summary(code)
                for code in _sequence_items(item, "MeasurementUnitsCodeSequence")
            )
            quantities = []
            for quantity in _sequence_items(item, "QuantityDefinitionSequence"):
                for sequence_name in (
                    "ConceptNameCodeSequence",
                    "ConceptCodeSequence",
                ):
                    for code in _sequence_items(quantity, sequence_name):
                        quantities.append(
                            f"{sequence_name}: {_code_summary(code)}"
                        )
            mapping = RWVMapping(
                suv_type=_mapping_suv_type(item),
                first_value_mapped=_normalized_value(
                    _value(item, "RealWorldValueFirstValueMapped")
                ),
                last_value_mapped=_normalized_value(
                    _value(item, "RealWorldValueLastValueMapped")
                ),
                slope=_normalized_value(_value(item, "RealWorldValueSlope")),
                intercept=_normalized_value(_value(item, "RealWorldValueIntercept")),
                measurement_units=units,
                quantity_definitions=tuple(quantities),
            )
            if all(mapping.signature != existing.signature for existing in self.mappings):
                self.mappings.append(mapping)

    def _collect_references(self, items: Sequence[Dataset]) -> None:
        pending = list(items)
        while pending:
            item = pending.pop(0)
            series_uid = _normalized_value(_value(item, "SeriesInstanceUID"))
            sop_uid = _normalized_value(_value(item, "ReferencedSOPInstanceUID"))
            if isinstance(series_uid, str) and series_uid not in self.referenced_series_uids:
                self.referenced_series_uids.append(series_uid)
            if isinstance(sop_uid, str) and sop_uid not in self.referenced_sop_instance_uids:
                self.referenced_sop_instance_uids.append(sop_uid)
            for sequence_name in (
                "ReferencedSeriesSequence",
                "ReferencedImageSequence",
                "ReferencedInstanceSequence",
                "ReferencedImageRealWorldValueMappingSequence",
            ):
                pending.extend(_sequence_items(item, sequence_name))


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
    pet_details: PTDetails | None = None
    rwv_details: RWVDetails | None = None
    source_paths: list[Path] = field(default_factory=list, repr=False)

    def add(
        self,
        dataset: Dataset,
        *,
        path: Path | None = None,
        include_pet_details: bool = False,
        retain_probe_instances: bool = False,
        retain_source_paths: bool = False,
    ) -> None:
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

        if include_pet_details:
            if self.pet_details is None:
                self.pet_details = PTDetails()
            if self.rwv_details is None:
                self.rwv_details = RWVDetails()
            self.pet_details.add(
                dataset, path=path, retain_instance=retain_probe_instances
            )
            self.rwv_details.add(dataset)
        if retain_source_paths and path is not None:
            self.source_paths.append(path)

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


def inventory_directory(
    source: Path,
    *,
    include_pet_details: bool = False,
    retain_probe_instances: bool = False,
    retain_source_paths: bool = False,
) -> InventoryResult:
    """Read allowlisted metadata from DICOM files beneath an external directory."""

    source = _validate_source(source)
    result = InventoryResult()
    include_pet_details = include_pet_details or retain_probe_instances
    read_tags = PET_READ_TAGS if include_pet_details else READ_TAGS

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
                    specific_tags=read_tags,
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
            series.add(
                dataset,
                path=path,
                include_pet_details=include_pet_details,
                retain_probe_instances=retain_probe_instances,
                retain_source_paths=retain_source_paths,
            )
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


def _format_observation(observation: FieldObservation) -> str:
    state = observation.state
    if not observation.values:
        return state

    if len(observation.values) == 1:
        value = _format_normalized(observation.values[0])
        if state == "PARTIAL":
            return f"{value} [PARTIAL; missing on {observation.missing_instances} instances]"
        return f"{value} [{state}]"

    rendered = [_format_normalized(value) for value in observation.values]
    preview = ", ".join(rendered[:3])
    if len(rendered) > 3:
        preview += f", ... ({len(rendered)} distinct values)"
    if observation.missing_instances:
        preview += f"; missing on {observation.missing_instances} instances"
    return f"{state} [{preview}]"


def _format_sequence_presence(observation: FieldObservation) -> str:
    if observation.state == "MISSING":
        return "MISSING"
    if observation.state == "PARTIAL":
        return (
            "PRESENT [PARTIAL; missing on "
            f"{observation.missing_instances} instances]"
        )
    return f"PRESENT [{observation.state}]"


def _series_with_modality(
    result: InventoryResult, modality: str
) -> list[tuple[str, SeriesInventory]]:
    matches = []
    for study_uid in sorted(result.studies):
        for series_uid in sorted(result.studies[study_uid]):
            series = result.studies[study_uid][series_uid]
            observed_modality = _normalized_value(series.metadata.get("Modality"))
            if isinstance(observed_modality, str) and observed_modality.upper() == modality:
                matches.append((study_uid, series))
    return matches


def _format_pet_series(study_uid: str, series: SeriesInventory) -> list[str]:
    details = series.pet_details
    if details is None:
        return []

    observations = details.observations
    lines = [
        f"Study: {study_uid}",
        f"  Series: {series.series_instance_uid}",
        "    SeriesDescription: "
        f"{_display_value(series.metadata.get('SeriesDescription'))}",
        f"    Modality: {_display_value(series.metadata.get('Modality'))}",
        f"    SOPClassUID: {_display_value(series.metadata.get('SOPClassUID'))}",
        f"    Units: {_format_observation(observations['Units'])}",
        f"    UnitsConsistency: {observations['Units'].state}",
        f"    SUVType: {_format_observation(observations['SUVType'])}",
        f"    RescaleSlope: {_format_observation(observations['RescaleSlope'])}",
        f"    RescaleIntercept: {_format_observation(observations['RescaleIntercept'])}",
        f"    CorrectedImage: {_format_observation(observations['CorrectedImage'])}",
        f"    DecayCorrection: {_format_observation(observations['DecayCorrection'])}",
        f"    PatientWeight: {_format_observation(observations['PatientWeight'])}",
        f"    PatientSize: {_format_observation(observations['PatientSize'])}",
        f"    PatientSex: {_format_observation(observations['PatientSex'])}",
        "    Timing:",
    ]
    for name in TIMING_FIELDS:
        lines.append(f"      {name}: {_format_observation(observations[name])}")

    lines.append(
        "    RadiopharmaceuticalInformationSequence: "
        f"{_format_sequence_presence(details.radiopharmaceutical_sequence)}"
    )
    for index, item in enumerate(details.radiopharmaceutical_items, start=1):
        lines.append(f"      Item {index}:")
        for name in RADIOPHARMACEUTICAL_FIELDS:
            value = item.values[name]
            rendered = "MISSING" if value is None else _format_normalized(value)
            lines.append(f"        {name}: {rendered}")
    return lines


def _format_uid_references(label: str, values: Sequence[str]) -> list[str]:
    if not values:
        return [f"    {label}: MISSING"]

    lines = [f"    {label}: {len(values)}"]
    for value in values[:5]:
        lines.append(f"      {value}")
    if len(values) > 5:
        lines.append(f"      ... {len(values) - 5} more")
    return lines


def _format_rwv_series(study_uid: str, series: SeriesInventory) -> list[str]:
    details = series.rwv_details
    if details is None:
        return []

    lines = [
        f"Study: {study_uid}",
        f"  Series: {series.series_instance_uid}",
        "    SeriesDescription: "
        f"{_display_value(series.metadata.get('SeriesDescription'))}",
        f"    SOPClassUID: {_display_value(series.metadata.get('SOPClassUID'))}",
    ]
    for name in RWV_SEQUENCE_FIELDS:
        lines.append(
            f"    {name}: {_format_sequence_presence(details.sequence_presence[name])}"
        )

    if details.mappings:
        for index, mapping in enumerate(details.mappings, start=1):
            lines.extend(
                [
                    f"    Mapping {index}:",
                    f"      SUVType: {mapping.suv_type}",
                    "      RealWorldValueFirstValueMapped: "
                    + (
                        "MISSING"
                        if mapping.first_value_mapped is None
                        else _format_normalized(mapping.first_value_mapped)
                    ),
                    "      RealWorldValueLastValueMapped: "
                    + (
                        "MISSING"
                        if mapping.last_value_mapped is None
                        else _format_normalized(mapping.last_value_mapped)
                    ),
                    "      RealWorldValueSlope: "
                    + (
                        "MISSING"
                        if mapping.slope is None
                        else _format_normalized(mapping.slope)
                    ),
                    "      RealWorldValueIntercept: "
                    + (
                        "MISSING"
                        if mapping.intercept is None
                        else _format_normalized(mapping.intercept)
                    ),
                ]
            )
            if mapping.measurement_units:
                for value in mapping.measurement_units:
                    lines.append(f"      MeasurementUnitsCodeSequence: {value}")
            else:
                lines.append("      MeasurementUnitsCodeSequence: MISSING")
            if mapping.quantity_definitions:
                for value in mapping.quantity_definitions:
                    lines.append(f"      QuantityDefinitionSequence: {value}")
            else:
                lines.append("      QuantityDefinitionSequence: MISSING")
    else:
        lines.append("    Mappings: MISSING")

    lines.extend(
        _format_uid_references(
            "ReferencedSourceSeriesInstanceUIDs",
            details.referenced_series_uids,
        )
    )
    lines.extend(
        _format_uid_references(
            "ReferencedSourceSOPInstanceUIDs",
            details.referenced_sop_instance_uids,
        )
    )
    return lines


def _combined_rescale_state(details: PTDetails) -> str:
    states = {
        details.observations["RescaleSlope"].state,
        details.observations["RescaleIntercept"].state,
    }
    for state in ("MISSING", "VARYING", "PARTIAL"):
        if state in states:
            return state
    return "CONSTANT"


def _has_decy_on_all_instances(observation: FieldObservation) -> bool:
    if observation.state != "CONSTANT" or observation.first_value is None:
        return False
    value = observation.first_value
    tokens = value if isinstance(value, tuple) else tuple(value.split("\\"))
    return "DECY" in {token.strip().upper() for token in tokens}


def _values_are_usable_numbers(
    observation: FieldObservation, *, nonzero: bool = False
) -> bool:
    if not observation.values:
        return False
    try:
        numbers = [float(_format_normalized(value)) for value in observation.values]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) and (not nonzero or value != 0) for value in numbers)


def _injection_timing_status(details: PTDetails) -> str:
    top_level = details.observations
    nested = details.radiopharmaceutical_observations
    if (
        top_level["RadiopharmaceuticalStartDateTime"].values
        or nested["RadiopharmaceuticalStartDateTime"].values
    ):
        return "present"
    if (
        top_level["RadiopharmaceuticalStartTime"].values
        or nested["RadiopharmaceuticalStartTime"].values
    ):
        return "partial"
    return "missing"


def _readiness(details: PTDetails) -> tuple[str, list[str]]:
    observations = details.observations
    radiopharmaceutical = details.radiopharmaceutical_observations
    injection_status = _injection_timing_status(details)

    essential = {
        "Units": observations["Units"],
        "RescaleSlope": observations["RescaleSlope"],
        "RescaleIntercept": observations["RescaleIntercept"],
        "PatientWeight": observations["PatientWeight"],
        "RadionuclideTotalDose": radiopharmaceutical["RadionuclideTotalDose"],
        "RadionuclideHalfLife": radiopharmaceutical["RadionuclideHalfLife"],
        "DecayCorrection": observations["DecayCorrection"],
    }
    missing = [name for name, value in essential.items() if not value.values]
    if injection_status == "missing":
        missing.append("InjectionDateTime")
    if missing:
        return "NO", ["missing essential inputs: " + ", ".join(missing)]

    review_reasons = []
    units = observations["Units"]
    if units.state != "CONSTANT":
        review_reasons.append(f"Units are {units.state}")
    elif str(units.first_value).upper() != "BQML":
        review_reasons.append("Units are not BQML")

    for name in ("RescaleSlope", "RescaleIntercept"):
        if observations[name].state != "CONSTANT":
            review_reasons.append(f"{name} is {observations[name].state}")
    if not _values_are_usable_numbers(observations["RescaleSlope"], nonzero=True):
        review_reasons.append("RescaleSlope is not a usable non-zero number")
    if not _values_are_usable_numbers(observations["RescaleIntercept"]):
        review_reasons.append("RescaleIntercept is not a usable number")

    for name in ("PatientWeight",):
        if observations[name].state != "CONSTANT":
            review_reasons.append(f"{name} is {observations[name].state}")
    for name in ("RadionuclideTotalDose", "RadionuclideHalfLife"):
        if radiopharmaceutical[name].state != "CONSTANT":
            review_reasons.append(f"{name} is {radiopharmaceutical[name].state}")

    if injection_status != "present":
        review_reasons.append("InjectionDateTime is partial")
    for name in TIMING_FIELDS:
        state = observations[name].state
        if state in {"VARYING", "PARTIAL"}:
            review_reasons.append(f"{name} is {state}")

    decay = observations["DecayCorrection"]
    if decay.state != "CONSTANT":
        review_reasons.append(f"DecayCorrection is {decay.state}")
    elif str(decay.first_value).upper() not in {"START", "ADMIN"}:
        review_reasons.append("DecayCorrection is unusual or ambiguous")

    if not _has_decy_on_all_instances(observations["CorrectedImage"]):
        review_reasons.append("CorrectedImage does not consistently contain DECY")

    return ("REVIEW", review_reasons) if review_reasons else ("YES", [])


def _presence_label(observation: FieldObservation) -> str:
    if not observation.values:
        return "missing"
    if observation.state == "CONSTANT":
        return "present"
    return f"present ({observation.state})"


def _format_quantitation_summary(series: SeriesInventory) -> list[str]:
    details = series.pet_details
    if details is None:
        return []

    observations = details.observations
    radiopharmaceutical = details.radiopharmaceutical_observations
    units = observations["Units"].first_value
    units_value = "MISSING" if units is None else _format_normalized(units)
    ready, reasons = _readiness(details)
    lines = [
        f"Series: {series.series_instance_uid}",
        "  QuantitationInputs:",
        f"    Units: {units_value}",
        f"    Rescale: {_combined_rescale_state(details)}",
        f"    PatientWeight: {_presence_label(observations['PatientWeight'])}",
        "    RadionuclideTotalDose: "
        f"{_presence_label(radiopharmaceutical['RadionuclideTotalDose'])}",
        "    RadionuclideHalfLife: "
        f"{_presence_label(radiopharmaceutical['RadionuclideHalfLife'])}",
        f"    InjectionDateTime: {_injection_timing_status(details)}",
        f"    DecayCorrection: {_format_observation(observations['DecayCorrection'])}",
        "    CorrectedImageContainsDECY: "
        f"{'yes' if _has_decy_on_all_instances(observations['CorrectedImage']) else 'no'}",
        f"    ReadyForSUVValidation: {ready}",
    ]
    for reason in reasons:
        lines.append(f"    ReadinessNote: {reason}")
    return lines


def format_pet_details(result: InventoryResult) -> str:
    """Format allowlisted PT and RWV metadata without calculating SUV."""

    pt_series = _series_with_modality(result, "PT")
    rwv_series = _series_with_modality(result, "RWV")
    lines = ["PET SERIES"]
    if pt_series:
        for study_uid, series in pt_series:
            lines.extend(_format_pet_series(study_uid, series))
    else:
        lines.append("No PT series found.")

    lines.extend(["", "RWV / SUV MAPPING SERIES"])
    if rwv_series:
        for study_uid, series in rwv_series:
            lines.extend(_format_rwv_series(study_uid, series))
    else:
        lines.append("No RWV series found.")

    lines.extend(["", "QUANTITATION INPUT SUMMARY"])
    if pt_series:
        for _study_uid, series in pt_series:
            lines.extend(_format_quantitation_summary(series))
    else:
        lines.append("No PT series available for metadata completeness assessment.")

    lines.extend(
        [
            "",
            "ReadyForSUVValidation reports metadata completeness only; it does not",
            "prove that SUV values are correct.",
        ]
    )
    return "\n".join(lines)


def _as_float(value: str | tuple[str, ...] | None) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _record_z(record: PTInstanceRecord) -> float | None:
    if record.image_position_patient is None or len(record.image_position_patient) < 3:
        return None
    return _as_float(record.image_position_patient[2])


def _instance_sort_key(record: PTInstanceRecord) -> tuple[Any, ...]:
    z_position = _record_z(record)
    if z_position is not None:
        return (0, z_position, record.sop_instance_uid)
    instance_number = _as_float(record.instance_number)
    if instance_number is not None:
        return (1, instance_number, record.sop_instance_uid)
    return (2, record.sop_instance_uid)


def _quantitative_pt_series(
    result: InventoryResult,
) -> list[tuple[str, SeriesInventory]]:
    matches = []
    for study_uid, series in _series_with_modality(result, "PT"):
        if series.pet_details is None:
            continue
        units = series.pet_details.observations["Units"]
        if (
            units.state == "CONSTANT"
            and str(units.first_value).upper() == "BQML"
            and series.pet_details.instances
        ):
            matches.append((study_uid, series))
    return matches


def _selected_probe_records(
    records: Sequence[PTInstanceRecord],
) -> list[tuple[PTInstanceRecord, tuple[str, ...]]]:
    ordered = sorted(records, key=_instance_sort_key)
    selected: dict[str, tuple[PTInstanceRecord, list[str]]] = {}

    def add(record: PTInstanceRecord, reason: str) -> None:
        existing = selected.get(record.sop_instance_uid)
        if existing is None:
            selected[record.sop_instance_uid] = (record, [reason])
        elif reason not in existing[1]:
            existing[1].append(reason)

    add(ordered[0], "first geometrical slice")
    add(ordered[len(ordered) // 2], "middle geometrical slice")
    add(ordered[-1], "last geometrical slice")

    by_acquisition_time: dict[str, list[PTInstanceRecord]] = {}
    for record in records:
        key = record.acquisition_time or "MISSING"
        by_acquisition_time.setdefault(key, []).append(record)
    for acquisition_time in sorted(by_acquisition_time):
        group = sorted(by_acquisition_time[acquisition_time], key=_instance_sort_key)
        add(
            group[len(group) // 2],
            f"representative of AcquisitionTime {acquisition_time}",
        )

    return [
        (record, tuple(reasons))
        for record, reasons in sorted(
            selected.values(), key=lambda item: _instance_sort_key(item[0])
        )
    ]


@dataclass
class PixelProbePoint:
    label: str
    row: int
    column: int
    stored_value: int | float


@dataclass
class PixelProbe:
    record: PTInstanceRecord
    reasons: tuple[str, ...]
    transfer_syntax_uid: str | None
    bits_allocated: str | None
    bits_stored: str | None
    high_bit: str | None
    pixel_representation: str | None
    minimum_stored_value: int | float | None = None
    maximum_stored_value: int | float | None = None
    pixels_outside_mapped_range: int | None = None
    total_pixels: int | None = None
    points: list[PixelProbePoint] = field(default_factory=list)
    error: str | None = None


def _read_pixel_probe(
    record: PTInstanceRecord,
    reasons: tuple[str, ...],
    mapped_range: tuple[float, float] | None,
) -> PixelProbe:
    try:
        dataset = pydicom.dcmread(record.path, force=False)
    except Exception as error:
        return PixelProbe(
            record=record,
            reasons=reasons,
            transfer_syntax_uid=record.transfer_syntax_uid,
            bits_allocated=record.bits_allocated,
            bits_stored=record.bits_stored,
            high_bit=record.high_bit,
            pixel_representation=record.pixel_representation,
            error=f"DICOM read failed: {type(error).__name__}: {error}",
        )

    transfer_syntax_uid = _transfer_syntax_uid(dataset)
    probe = PixelProbe(
        record=record,
        reasons=reasons,
        transfer_syntax_uid=transfer_syntax_uid,
        bits_allocated=_string_value(dataset, "BitsAllocated"),
        bits_stored=_string_value(dataset, "BitsStored"),
        high_bit=_string_value(dataset, "HighBit"),
        pixel_representation=_string_value(dataset, "PixelRepresentation"),
    )
    try:
        pixels = np.asarray(dataset.pixel_array)
        if pixels.ndim != 2:
            raise ValueError(
                f"expected one two-dimensional frame, decoded shape is {pixels.shape}"
            )
        probe.minimum_stored_value = pixels.min().item()
        probe.maximum_stored_value = pixels.max().item()
        probe.total_pixels = int(pixels.size)
        if mapped_range is None:
            raise ValueError("SUVbw RWV declared mapped range is unavailable")

        first_mapped, last_mapped = mapped_range
        valid_mask = (pixels >= first_mapped) & (pixels <= last_mapped)
        probe.pixels_outside_mapped_range = int(
            np.count_nonzero(~valid_mask)
        )
        flat_pixels = pixels.reshape(-1)
        positive_valid_indices = np.flatnonzero(
            valid_mask.reshape(-1) & (flat_pixels > 0)
        )
        if positive_valid_indices.size == 0:
            raise ValueError("no positive stored pixels fall within the RWV range")

        selected_indices: set[int] = set()

        def add_point(label: str, target: float) -> None:
            candidates = np.asarray(
                [
                    int(index)
                    for index in positive_valid_indices
                    if int(index) not in selected_indices
                ],
                dtype=np.int64,
            )
            if candidates.size == 0:
                return
            distances = np.abs(flat_pixels[candidates].astype(np.float64) - target)
            flat_index = int(candidates[int(np.argmin(distances))])
            selected_indices.add(flat_index)
            row, column = np.unravel_index(flat_index, pixels.shape)
            probe.points.append(
                PixelProbePoint(
                    label=label,
                    row=int(row),
                    column=int(column),
                    stored_value=flat_pixels[flat_index].item(),
                )
            )

        positive_values = flat_pixels[positive_valid_indices].astype(np.float64)
        add_point("maximum stored value within RWV range", float(positive_values.max()))
        add_point("approximately median positive stored value", float(np.median(positive_values)))
        add_point("ordinary positive stored value", float(np.quantile(positive_values, 0.25)))
    except Exception as error:
        probe.error = f"{type(error).__name__}: {error}"
    return probe


def _format_number(value: float | int) -> str:
    return f"{value:.12g}"


def _format_range(values: Sequence[float]) -> str:
    if not values:
        return "MISSING"
    return f"{_format_number(min(values))} to {_format_number(max(values))}"


def _format_acquisition_groups(records: Sequence[PTInstanceRecord]) -> list[str]:
    groups: dict[str, list[PTInstanceRecord]] = {}
    for record in records:
        groups.setdefault(record.acquisition_time or "MISSING", []).append(record)

    lines = ["  ACQUISITION TIME / BED-POSITION GROUPS"]
    for acquisition_time in sorted(groups):
        group = groups[acquisition_time]
        z_positions = [
            value for record in group if (value := _record_z(record)) is not None
        ]
        slopes = [
            value
            for record in group
            if (value := _as_float(record.rescale_slope)) is not None
        ]
        lines.extend(
            [
                f"    AcquisitionTime: {acquisition_time}",
                f"      NumberOfInstances: {len(group)}",
                f"      ImagePositionPatientZRange: {_format_range(z_positions)}",
                f"      RescaleSlopeRange: {_format_range(slopes)}",
            ]
        )
    return lines


@dataclass(frozen=True)
class MetadataSUVFactor:
    value: float | None
    reason: str | None
    basis: str | None = None
    injection_datetime: datetime | None = None
    reference_datetime: datetime | None = None
    elapsed_seconds: float | None = None
    decay_corrected_dose_bq: float | None = None


def _parse_timezone_offset(value: str | None) -> timezone | None:
    if not value or not re.fullmatch(r"[+-]\d{4}", value):
        return None
    sign = 1 if value[0] == "+" else -1
    hours = int(value[1:3])
    minutes = int(value[3:5])
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _parse_dicom_datetime(value: str, default_tz: timezone | None) -> datetime:
    match = re.fullmatch(r"(\d{14})(?:\.(\d{1,6})\d*)?([+-]\d{4})?", value)
    if match is None:
        raise ValueError("requires a complete YYYYMMDDHHMMSS DICOM datetime")
    parsed = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    fraction = (match.group(2) or "").ljust(6, "0")
    parsed = parsed.replace(microsecond=int(fraction or "0"))
    explicit_tz = _parse_timezone_offset(match.group(3))
    return parsed.replace(tzinfo=explicit_tz or default_tz)


def _parse_series_datetime(
    date_value: str, time_value: str, default_tz: timezone | None
) -> datetime:
    match = re.fullmatch(r"(\d{6})(?:\.(\d{1,6})\d*)?", time_value)
    if not re.fullmatch(r"\d{8}", date_value) or match is None:
        raise ValueError("requires complete SeriesDate and SeriesTime")
    parsed = datetime.strptime(date_value + match.group(1), "%Y%m%d%H%M%S")
    fraction = (match.group(2) or "").ljust(6, "0")
    return parsed.replace(microsecond=int(fraction or "0"), tzinfo=default_tz)


def _constant_observation_value(
    observation: FieldObservation,
) -> str | None:
    value = observation.first_value
    if observation.state != "CONSTANT" or not isinstance(value, str):
        return None
    return value


def _metadata_suvbw_factor(details: PTDetails) -> MetadataSUVFactor:
    observations = details.observations
    radiopharmaceutical = details.radiopharmaceutical_observations
    required = {
        "PatientWeight": observations["PatientWeight"],
        "RadionuclideTotalDose": radiopharmaceutical["RadionuclideTotalDose"],
        "RadionuclideHalfLife": radiopharmaceutical["RadionuclideHalfLife"],
        "DecayCorrection": observations["DecayCorrection"],
    }
    invalid = [name for name, observation in required.items() if observation.state != "CONSTANT"]
    if invalid:
        return MetadataSUVFactor(
            None,
            "required fields are not constant: " + ", ".join(invalid),
        )

    weight_kg = _as_float(required["PatientWeight"].first_value)
    injected_dose_bq = _as_float(required["RadionuclideTotalDose"].first_value)
    half_life_seconds = _as_float(required["RadionuclideHalfLife"].first_value)
    if (
        weight_kg is None
        or injected_dose_bq is None
        or half_life_seconds is None
        or weight_kg <= 0
        or injected_dose_bq <= 0
        or half_life_seconds <= 0
    ):
        return MetadataSUVFactor(None, "weight, dose, or half-life is not positive numeric metadata")

    decay_correction = str(required["DecayCorrection"].first_value).upper()
    if decay_correction == "ADMIN":
        return MetadataSUVFactor(
            value=weight_kg * 1000.0 / injected_dose_bq,
            reason=None,
            basis="DecayCorrection=ADMIN; dose referenced to administration time",
            elapsed_seconds=0.0,
            decay_corrected_dose_bq=injected_dose_bq,
        )
    if decay_correction != "START":
        return MetadataSUVFactor(
            None,
            f"DecayCorrection={decay_correction or 'MISSING'} does not define a safe single START/ADMIN factor",
        )

    injection = radiopharmaceutical["RadiopharmaceuticalStartDateTime"]
    series_date = observations["SeriesDate"]
    series_time = observations["SeriesTime"]
    injection_value = _constant_observation_value(injection)
    date_value = _constant_observation_value(series_date)
    time_value = _constant_observation_value(series_time)
    if not injection_value or not date_value or not time_value:
        return MetadataSUVFactor(
            None,
            "DecayCorrection=START requires constant injection datetime and SeriesDate/SeriesTime",
        )

    timezone_value = _constant_observation_value(
        observations["TimezoneOffsetFromUTC"]
    )
    default_tz = _parse_timezone_offset(timezone_value)
    try:
        injection_datetime = _parse_dicom_datetime(injection_value, default_tz)
        reference_datetime = _parse_series_datetime(
            date_value, time_value, default_tz
        )
        if (injection_datetime.tzinfo is None) != (reference_datetime.tzinfo is None):
            raise ValueError("injection and reference time zones are inconsistent")
        elapsed_seconds = (reference_datetime - injection_datetime).total_seconds()
    except ValueError as error:
        return MetadataSUVFactor(None, str(error))
    if elapsed_seconds < 0:
        return MetadataSUVFactor(None, "decay reference precedes injection datetime")

    decay_corrected_dose = injected_dose_bq * math.exp(
        -math.log(2.0) * elapsed_seconds / half_life_seconds
    )
    timezone_note = (
        f"TimezoneOffsetFromUTC={timezone_value}"
        if default_tz is not None
        else "no timezone offset; values treated as the same local DICOM clock"
    )
    return MetadataSUVFactor(
        value=weight_kg * 1000.0 / decay_corrected_dose,
        reason=None,
        basis=(
            "DecayCorrection=START; SeriesDate+SeriesTime used as the DICOM PET "
            f"series reference time; {timezone_note}"
        ),
        injection_datetime=injection_datetime,
        reference_datetime=reference_datetime,
        elapsed_seconds=elapsed_seconds,
        decay_corrected_dose_bq=decay_corrected_dose,
    )


def _rwv_mappings_for_pt_series(
    result: InventoryResult, study_uid: str, series_uid: str
) -> list[RWVMapping]:
    mappings = []
    for rwv_study_uid, rwv_series in _series_with_modality(result, "RWV"):
        details = rwv_series.rwv_details
        if details is None or rwv_study_uid != study_uid:
            continue
        if details.referenced_series_uids and series_uid not in details.referenced_series_uids:
            continue
        for mapping in details.mappings:
            if mapping.signature not in {item.signature for item in mappings}:
                mappings.append(mapping)
    return mappings


def _mapping_number(value: str | tuple[str, ...] | None) -> float | None:
    return _as_float(value)


def _format_metadata_factor(factor: MetadataSUVFactor) -> list[str]:
    lines = ["  METADATA-DERIVED BQML TO SUVbw FACTOR"]
    if factor.value is None:
        lines.append("    MetadataDerivedBQMLToSUVbwFactor: NOT CALCULATED")
        lines.append(f"    Reason: {factor.reason}")
        return lines

    lines.append(
        "    MetadataDerivedBQMLToSUVbwFactor: "
        f"{_format_number(factor.value)}"
    )
    if factor.basis:
        lines.append(f"    TimingBasis: {factor.basis}")
    if factor.injection_datetime:
        lines.append(
            f"    InjectionDateTimeUsed: {factor.injection_datetime.isoformat()}"
        )
    if factor.reference_datetime:
        lines.append(
            f"    DecayReferenceDateTimeUsed: {factor.reference_datetime.isoformat()}"
        )
    if factor.elapsed_seconds is not None:
        lines.append(
            f"    InjectionToReferenceSeconds: {_format_number(factor.elapsed_seconds)}"
        )
    if factor.decay_corrected_dose_bq is not None:
        lines.append(
            "    DecayCorrectedInjectedActivityBq: "
            f"{_format_number(factor.decay_corrected_dose_bq)}"
        )
    return lines


def _suvbw_mapping(mappings: Sequence[RWVMapping]) -> RWVMapping | None:
    return next((mapping for mapping in mappings if mapping.suv_type == "SUVbw"), None)


def _mapping_range(mapping: RWVMapping | None) -> tuple[float, float] | None:
    if mapping is None:
        return None
    first_mapped = _mapping_number(mapping.first_value_mapped)
    last_mapped = _mapping_number(mapping.last_value_mapped)
    if first_mapped is None or last_mapped is None or first_mapped > last_mapped:
        return None
    return first_mapped, last_mapped


def _format_factor_vs_rwv(
    factor: MetadataSUVFactor, mapping: RWVMapping | None
) -> list[str]:
    lines = ["  RWVSlopeVsMetadataBQMLFactor"]
    rwv_slope = _mapping_number(mapping.slope) if mapping else None
    if factor.value is None or rwv_slope is None:
        lines.append("    Result: NOT ASSESSED")
        lines.append("    Reason: metadata factor or SUVbw RWV slope is unavailable")
        return lines

    absolute_difference = abs(rwv_slope - factor.value)
    relative_difference = (
        absolute_difference / abs(factor.value)
        if factor.value != 0
        else math.inf
    )
    percentage_difference = relative_difference * 100.0
    lines.extend(
        [
            f"    RWVSUVbwSlope: {_format_number(rwv_slope)}",
            "    MetadataDerivedBQMLToSUVbwFactor: "
            f"{_format_number(factor.value)}",
            f"    AbsoluteDifference: {_format_number(absolute_difference)}",
            f"    RelativeDifference: {_format_number(relative_difference)}",
            f"    PercentageDifference: {_format_number(percentage_difference)}%",
            "    Result: "
            + ("MATCH" if percentage_difference <= 0.1 else "REVIEW")
            + " [0.1% tolerance]",
        ]
    )
    return lines


def _format_direct_stored_compatibility(
    details: PTDetails,
    factor: MetadataSUVFactor,
    mapping: RWVMapping | None,
) -> list[str]:
    lines = ["  DirectStoredValueRWVInterpretation:"]
    rwv_slope = _mapping_number(mapping.slope) if mapping else None
    rescale_slopes = [
        value
        for item in details.observations["RescaleSlope"].values
        if (value := _as_float(item)) is not None
    ]
    if factor.value is None or rwv_slope is None or not rescale_slopes:
        lines.append("    Status: NOT_COMPATIBLE")
        lines.append("    Reason: required comparison factors are unavailable")
        return lines

    expected_slopes = [value * factor.value for value in rescale_slopes]
    percentage_differences = [
        abs(value - rwv_slope) / abs(rwv_slope) * 100.0
        if rwv_slope != 0
        else math.inf
        for value in expected_slopes
    ]
    compatible = max(percentage_differences) <= 0.1
    lines.extend(
        [
            f"    Status: {'COMPATIBLE' if compatible else 'NOT_COMPATIBLE'}",
            f"    DistinctInstanceRescaleSlopes: {len(rescale_slopes)}",
            "    ExpectedStoredToSUVbwSlopeRange: "
            f"{_format_range(expected_slopes)}",
            f"    GlobalRWVSUVbwSlope: {_format_number(rwv_slope)}",
            "    PercentageDifferenceRange: "
            f"{_format_range(percentage_differences)}%",
        ]
    )
    if not compatible:
        lines.append(
            "    Interpretation: the global RWV slope does not represent the "
            "instance-specific stored-value-to-SUVbw slopes. This is descriptive, "
            "not a corrupt-study warning."
        )
    return lines


def _dicom_time_seconds(value: str) -> float | None:
    match = re.fullmatch(r"(\d{2})(\d{2})(\d{2})(?:\.(\d+))?", value)
    if match is None:
        return None
    hours, minutes, seconds = (int(match.group(index)) for index in range(1, 4))
    if hours > 23 or minutes > 59 or seconds > 60:
        return None
    fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds + fraction


def _format_series_time_check(
    details: PTDetails, records: Sequence[PTInstanceRecord]
) -> list[str]:
    lines = ["  SERIES / ACQUISITION TIME CONSISTENCY"]
    series_time_value = _constant_observation_value(
        details.observations["SeriesTime"]
    )
    series_date_value = _constant_observation_value(
        details.observations["SeriesDate"]
    )
    acquisition_date_value = _constant_observation_value(
        details.observations["AcquisitionDate"]
    )
    acquisition_times = sorted(
        {
            record.acquisition_time
            for record in records
            if record.acquisition_time is not None
        }
    )
    parsed_series_time = (
        _dicom_time_seconds(series_time_value) if series_time_value else None
    )
    parsed_acquisition_times = [
        value
        for item in acquisition_times
        if (value := _dicom_time_seconds(item)) is not None
    ]
    lines.extend(
        [
            f"    SeriesTime: {series_time_value or 'MISSING'}",
            "    EarliestAcquisitionTime: "
            + (acquisition_times[0] if acquisition_times else "MISSING"),
        ]
    )
    if (
        parsed_series_time is None
        or len(parsed_acquisition_times) != len(acquisition_times)
        or not parsed_acquisition_times
        or not series_date_value
        or series_date_value != acquisition_date_value
    ):
        lines.extend(
            [
                "    SeriesTimeEarlierThanOrEqualToAllAcquisitionTimes: NOT ASSESSED",
                "    SeriesTimeMatchesEarliestAcquisitionTime: NOT ASSESSED",
                "    TimingCheckType: descriptive only",
            ]
        )
        return lines

    earliest = min(parsed_acquisition_times)
    lines.extend(
        [
            "    SeriesTimeEarlierThanOrEqualToAllAcquisitionTimes: "
            + ("YES" if parsed_series_time <= earliest else "NO"),
            "    SeriesTimeMatchesEarliestAcquisitionTime: "
            + ("YES" if parsed_series_time == earliest else "NO"),
            "    TimingCheckType: descriptive only",
        ]
    )
    return lines


def _point_quantitation(
    point: PixelProbePoint,
    record: PTInstanceRecord,
    factor: MetadataSUVFactor,
    rwv_slope: float | None,
) -> tuple[float, float, float, float] | None:
    instance_slope = _as_float(record.rescale_slope)
    instance_intercept = _as_float(record.rescale_intercept)
    if (
        instance_slope is None
        or instance_intercept is None
        or factor.value is None
        or rwv_slope is None
    ):
        return None
    bqml = float(point.stored_value) * instance_slope + instance_intercept
    suvbw = bqml * factor.value
    suvbw_using_rwv = bqml * rwv_slope
    return bqml, suvbw, suvbw_using_rwv, abs(suvbw - suvbw_using_rwv)


def _format_pixel_probe(
    probe: PixelProbe,
    mapping: RWVMapping | None,
    factor: MetadataSUVFactor,
) -> list[str]:
    record = probe.record
    lines = [
        f"    SOPInstanceUID: {record.sop_instance_uid}",
        f"      SelectionReason: {', '.join(probe.reasons)}",
        f"      InstanceNumber: {record.instance_number or 'MISSING'}",
        "      ImagePositionPatient: "
        + (
            "MISSING"
            if record.image_position_patient is None
            else "\\".join(record.image_position_patient)
        ),
        f"      SliceLocation: {record.slice_location or 'MISSING'}",
        f"      AcquisitionTime: {record.acquisition_time or 'MISSING'}",
        f"      RescaleSlope: {record.rescale_slope or 'MISSING'}",
        f"      RescaleIntercept: {record.rescale_intercept or 'MISSING'}",
        f"      BitsAllocated: {probe.bits_allocated or 'MISSING'}",
        f"      BitsStored: {probe.bits_stored or 'MISSING'}",
        f"      HighBit: {probe.high_bit or 'MISSING'}",
        f"      PixelRepresentation: {probe.pixel_representation or 'MISSING'}",
        f"      TransferSyntaxUID: {probe.transfer_syntax_uid or 'MISSING'}",
    ]
    if probe.error:
        lines.append(f"      PixelDecodeError: {probe.error}")
        return lines

    mapped_range = _mapping_range(mapping)
    lines.extend(
        [
            f"      MinStoredPixelValue: {_format_number(probe.minimum_stored_value)}",
            f"      MaxStoredPixelValue: {_format_number(probe.maximum_stored_value)}",
            "      RealWorldValueFirstValueMapped: "
            + ("MISSING" if mapped_range is None else _format_number(mapped_range[0])),
            "      RealWorldValueLastValueMapped: "
            + ("MISSING" if mapped_range is None else _format_number(mapped_range[1])),
            "      PixelsOutsideDeclaredMappingRange: "
            + (
                "NOT ASSESSED"
                if probe.pixels_outside_mapped_range is None
                else str(probe.pixels_outside_mapped_range)
            ),
            f"      TotalPixels: {probe.total_pixels}",
        ]
    )
    rwv_slope = _mapping_number(mapping.slope) if mapping else None
    for point in probe.points:
        values = _point_quantitation(point, record, factor, rwv_slope)
        lines.extend(
            [
                f"      ProbePoint: {point.label}",
                f"        Row: {point.row}",
                f"        Column: {point.column}",
                f"        StoredValue: {_format_number(point.stored_value)}",
                f"        InstanceRescaleSlope: {record.rescale_slope}",
                f"        InstanceRescaleIntercept: {record.rescale_intercept}",
                "        StoredValueWithinDeclaredMappingRange: YES",
            ]
        )
        if values is None:
            lines.append("        CandidateSUVbwTransformation: NOT CALCULATED")
            continue
        bqml, suvbw, suvbw_using_rwv, absolute_difference = values
        percentage_difference = (
            absolute_difference / abs(suvbw) * 100.0 if suvbw != 0 else 0.0
        )
        lines.extend(
            [
                f"        BQML: {_format_number(bqml)}",
                "        MetadataDerivedBQMLToSUVbwFactor: "
                f"{_format_number(factor.value)}",
                f"        SUVbw: {_format_number(suvbw)}",
                "        SUVbwUsingRWVFactor: "
                f"{_format_number(suvbw_using_rwv)}",
                "        SUVbwAbsoluteDifference: "
                f"{_format_number(absolute_difference)}",
                "        SUVbwPercentageDifference: "
                f"{_format_number(percentage_difference)}%",
            ]
        )
    return lines


def _format_probe_agreement(
    probes: Sequence[PixelProbe],
    mapping: RWVMapping | None,
    factor: MetadataSUVFactor,
) -> list[str]:
    lines = ["  MetadataVsRWVSUVAgreement:"]
    rwv_slope = _mapping_number(mapping.slope) if mapping else None
    differences = []
    percentage_differences = []
    for probe in probes:
        for point in probe.points:
            values = _point_quantitation(point, probe.record, factor, rwv_slope)
            if values is None:
                continue
            _bqml, suvbw, _suvbw_using_rwv, difference = values
            differences.append(difference)
            percentage_differences.append(
                difference / abs(suvbw) * 100.0 if suvbw != 0 else 0.0
            )
    if not differences:
        lines.append("    Status: REVIEW")
        lines.append("    Reason: no valid quantitative probe comparisons")
        return lines

    maximum_percentage = max(percentage_differences)
    lines.extend(
        [
            f"    Status: {'MATCH' if maximum_percentage <= 0.1 else 'REVIEW'}",
            f"    NumberOfProbePoints: {len(differences)}",
            f"    MaximumAbsoluteDifference: {_format_number(max(differences))}",
            f"    MaximumPercentageDifference: {_format_number(maximum_percentage)}%",
            "    MatchTolerance: 0.1%",
        ]
    )
    return lines


def format_pet_pixel_probe(result: InventoryResult) -> str:
    """Decode representative PT slices and expose quantitation calculations."""

    quantitative_series = _quantitative_pt_series(result)
    lines = ["PET PIXEL-VALUE / QUANTITATION PROBE"]
    if not quantitative_series:
        lines.append("No quantitative PT series with constant Units=BQML found.")
        return "\n".join(lines)

    for study_uid, series in quantitative_series:
        details = series.pet_details
        if details is None:
            continue
        mappings = _rwv_mappings_for_pt_series(
            result, study_uid, series.series_instance_uid
        )
        suvbw_mapping = _suvbw_mapping(mappings)
        mapped_range = _mapping_range(suvbw_mapping)
        factor = _metadata_suvbw_factor(details)
        lines.extend(
            [
                f"Study: {study_uid}",
                f"  QuantitativePTSeries: {series.series_instance_uid}",
                "  SeriesDescription: "
                f"{_display_value(series.metadata.get('SeriesDescription'))}",
                f"  NumberOfInstances: {len(details.instances)}",
            ]
        )
        lines.extend(_format_acquisition_groups(details.instances))
        lines.extend(_format_series_time_check(details, details.instances))
        lines.extend(_format_metadata_factor(factor))
        lines.extend(_format_factor_vs_rwv(factor, suvbw_mapping))
        lines.extend(
            _format_direct_stored_compatibility(details, factor, suvbw_mapping)
        )
        lines.extend(
            [
                "  CandidateSUVbwTransformation:",
                "    BQML = StoredValue * InstanceRescaleSlope + InstanceRescaleIntercept",
                "    SUVbw = BQML * MetadataDerivedBQMLToSUVbwFactor",
                "    Status: candidate quantitative transformation; not fully validated",
            ]
        )
        lines.append("  RWV MAPPINGS")
        if mappings:
            for index, mapping in enumerate(mappings, start=1):
                lines.extend(
                    [
                        f"    Mapping {index}:",
                        f"      SUVType: {mapping.suv_type}",
                        "      RealWorldValueFirstValueMapped: "
                        + (
                            "MISSING"
                            if mapping.first_value_mapped is None
                            else _format_normalized(mapping.first_value_mapped)
                        ),
                        "      RealWorldValueLastValueMapped: "
                        + (
                            "MISSING"
                            if mapping.last_value_mapped is None
                            else _format_normalized(mapping.last_value_mapped)
                        ),
                        "      RealWorldValueSlope: "
                        + (
                            "MISSING"
                            if mapping.slope is None
                            else _format_normalized(mapping.slope)
                        ),
                        "      RealWorldValueIntercept: "
                        + (
                            "MISSING"
                            if mapping.intercept is None
                            else _format_normalized(mapping.intercept)
                        ),
                    ]
                )
        else:
            lines.append("    No referenced RWV mappings found.")

        lines.append("  SELECTED PIXEL PROBES")
        probes = []
        for record, reasons in _selected_probe_records(details.instances):
            probe = _read_pixel_probe(record, reasons, mapped_range)
            probes.append(probe)
            lines.extend(_format_pixel_probe(probe, suvbw_mapping, factor))
        lines.extend(_format_probe_agreement(probes, suvbw_mapping, factor))

    lines.extend(
        [
            "",
            "CandidateSUVbwTransformation is not yet fully validated.",
            "No complete SUV volume or NIfTI output was generated.",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect DICOM studies and series with read-only operations."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="source DICOM directory located outside this repository",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--pet-details",
        action="store_true",
        help="show allowlisted PT and RWV metadata for future SUV validation",
    )
    modes.add_argument(
        "--pet-pixel-probe",
        action="store_true",
        help="decode representative PT slices and show quantitation calculations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = inventory_directory(
            args.source,
            include_pet_details=args.pet_details or args.pet_pixel_probe,
            retain_probe_instances=args.pet_pixel_probe,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.pet_pixel_probe:
        print(format_pet_pixel_probe(result))
    elif args.pet_details:
        print(format_pet_details(result))
    else:
        print(format_inventory(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
