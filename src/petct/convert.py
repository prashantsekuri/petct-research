"""Create verified derived PET SUVbw and CT NIfTI volumes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pydicom

from . import inventory


PET_DESCRIPTION = "PET Q CLEAR"
CT_DESCRIPTION = "WB CECT"
OUTPUT_FILENAMES = {
    "pet_nifti": "PET_SUVbw.nii.gz",
    "ct_nifti": "CT_WB_CECT.nii.gz",
    "pet_provenance": "pet_provenance.json",
    "ct_provenance": "ct_provenance.json",
}
HEADER_TAGS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "Modality",
    "Units",
    "Rows",
    "Columns",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "SliceThickness",
    "RescaleSlope",
    "RescaleIntercept",
    "SamplesPerPixel",
    "NumberOfFrames",
)
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
ORIENTATION_TOLERANCE = 1e-5
POSITION_TOLERANCE_MM = 1e-3
SPACING_RELATIVE_TOLERANCE = 1e-3
FACTOR_MATCH_PERCENT_TOLERANCE = 0.1


class ConversionError(RuntimeError):
    """Raised when conversion cannot proceed without unsafe assumptions."""


@dataclass(frozen=True)
class SliceHeader:
    path: Path
    study_instance_uid: str
    series_instance_uid: str
    sop_instance_uid: str
    rows: int
    columns: int
    image_orientation_patient: np.ndarray
    image_position_patient: np.ndarray
    pixel_spacing: np.ndarray
    slice_thickness: float
    rescale_slope: float
    rescale_intercept: float
    samples_per_pixel: int
    number_of_frames: int
    transfer_syntax_uid: str


@dataclass(frozen=True)
class Geometry:
    headers: tuple[SliceHeader, ...]
    original_array_shape: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    original_lps_affine: np.ndarray
    original_ras_affine: np.ndarray
    original_orientation_codes: tuple[str, str, str]
    physical_bounds_ras_mm: tuple[tuple[float, float, float], tuple[float, float, float]]
    maximum_slice_position_error_mm: float
    maximum_in_plane_drift_mm: float
    measured_slice_spacing_range_mm: tuple[float, float]
    affine_semantics_checks: dict[str, bool]


@dataclass(frozen=True)
class VolumeStatistics:
    minimum: float
    maximum: float
    mean: float
    finite_voxels: int
    non_finite_voxels: int
    out_of_rwv_range_voxels: int | None = None
    out_of_rwv_range_affected_slices: int | None = None
    out_of_rwv_range_original_voxel_bounds: (
        tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ) = None
    out_of_rwv_range_physical_bounds_ras_mm: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None


@dataclass(frozen=True)
class CanonicalVolume:
    output_array_shape: tuple[int, int, int]
    canonical_ras_affine: np.ndarray
    canonical_orientation_codes: tuple[str, str, str]
    axes_permuted: bool
    axes_flipped: bool
    orientation_transform: np.ndarray
    physical_bounds_ras_mm: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class WrittenVolume:
    kind: str
    path: Path
    geometry: Geometry
    canonical: CanonicalVolume
    statistics: VolumeStatistics
    sha256: str
    size_bytes: int
    post_write_checks: dict[str, bool]


@dataclass(frozen=True)
class ConversionReport:
    pet: WrittenVolume
    ct: WrittenVolume
    physical_bounds_overlap: bool


def _safe_error(error: Exception, path: Path) -> str:
    return f"{type(error).__name__}: {error}".replace(str(path), "<source file>")


def _required_text(dataset: pydicom.dataset.Dataset, keyword: str) -> str:
    value = dataset.get(keyword)
    if value is None or value == "":
        raise ConversionError(f"required DICOM field {keyword} is missing")
    return str(value)


def _required_float(dataset: pydicom.dataset.Dataset, keyword: str) -> float:
    try:
        value = float(_required_text(dataset, keyword))
    except ValueError as error:
        raise ConversionError(f"DICOM field {keyword} is not numeric") from error
    if not math.isfinite(value):
        raise ConversionError(f"DICOM field {keyword} is not finite")
    return value


def _required_float_vector(
    dataset: pydicom.dataset.Dataset, keyword: str, length: int
) -> np.ndarray:
    value = dataset.get(keyword)
    if value is None or len(value) != length:
        raise ConversionError(
            f"required DICOM field {keyword} must contain {length} values"
        )
    try:
        vector = np.asarray([float(item) for item in value], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ConversionError(f"DICOM field {keyword} is not numeric") from error
    if not np.all(np.isfinite(vector)):
        raise ConversionError(f"DICOM field {keyword} is not finite")
    return vector


def _read_slice_headers(series: inventory.SeriesInventory) -> list[SliceHeader]:
    if not series.source_paths:
        raise ConversionError("selected series has no retained source instances")

    headers = []
    for path in series.source_paths:
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=HEADER_TAGS,
                force=False,
            )
            file_meta = getattr(dataset, "file_meta", None)
            transfer_syntax = (
                str(file_meta.TransferSyntaxUID)
                if file_meta is not None and "TransferSyntaxUID" in file_meta
                else "MISSING"
            )
            headers.append(
                SliceHeader(
                    path=path,
                    study_instance_uid=_required_text(dataset, "StudyInstanceUID"),
                    series_instance_uid=_required_text(dataset, "SeriesInstanceUID"),
                    sop_instance_uid=_required_text(dataset, "SOPInstanceUID"),
                    rows=int(_required_text(dataset, "Rows")),
                    columns=int(_required_text(dataset, "Columns")),
                    image_orientation_patient=_required_float_vector(
                        dataset, "ImageOrientationPatient", 6
                    ),
                    image_position_patient=_required_float_vector(
                        dataset, "ImagePositionPatient", 3
                    ),
                    pixel_spacing=_required_float_vector(dataset, "PixelSpacing", 2),
                    slice_thickness=_required_float(dataset, "SliceThickness"),
                    rescale_slope=_required_float(dataset, "RescaleSlope"),
                    rescale_intercept=_required_float(dataset, "RescaleIntercept"),
                    samples_per_pixel=int(
                        _required_text(dataset, "SamplesPerPixel")
                    ),
                    number_of_frames=int(dataset.get("NumberOfFrames", 1)),
                    transfer_syntax_uid=transfer_syntax,
                )
            )
        except ConversionError:
            raise
        except Exception as error:
            raise ConversionError(
                "failed to read a selected series header: " + _safe_error(error, path)
            ) from error
    return headers


def _one_value(values: set[Any], name: str) -> Any:
    if len(values) != 1:
        raise ConversionError(f"{name} is inconsistent across the selected series")
    return next(iter(values))


def _physical_bounds(
    shape: Sequence[int], affine: np.ndarray
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    corners = []
    for axis_0 in (0, shape[0] - 1):
        for axis_1 in (0, shape[1] - 1):
            for axis_2 in (0, shape[2] - 1):
                point = affine @ np.asarray(
                    [axis_0, axis_1, axis_2, 1.0], dtype=np.float64
                )
                corners.append(point[:3])
    corner_array = np.asarray(corners)
    return (
        tuple(float(value) for value in corner_array.min(axis=0)),
        tuple(float(value) for value in corner_array.max(axis=0)),
    )


def _validate_geometry(headers: list[SliceHeader], label: str) -> Geometry:
    if len(headers) < 2:
        raise ConversionError(f"{label} series requires at least two slices")

    if len({header.sop_instance_uid for header in headers}) != len(headers):
        raise ConversionError(f"{label} series contains duplicate SOP Instance UIDs")
    rows = _one_value({header.rows for header in headers}, f"{label} Rows")
    columns = _one_value(
        {header.columns for header in headers}, f"{label} Columns"
    )
    if rows <= 0 or columns <= 0:
        raise ConversionError(f"{label} rows and columns must be positive")
    if any(header.samples_per_pixel != 1 for header in headers):
        raise ConversionError(f"{label} series must have SamplesPerPixel=1")
    if any(header.number_of_frames != 1 for header in headers):
        raise ConversionError(f"{label} multi-frame instances are not supported")

    reference_iop = headers[0].image_orientation_patient
    dicom_row_direction = reference_iop[:3]
    dicom_column_direction = reference_iop[3:]
    if not math.isclose(
        float(np.linalg.norm(dicom_row_direction)), 1.0, abs_tol=ORIENTATION_TOLERANCE
    ):
        raise ConversionError(f"{label} DICOM row direction is not unit length")
    if not math.isclose(
        float(np.linalg.norm(dicom_column_direction)),
        1.0,
        abs_tol=ORIENTATION_TOLERANCE,
    ):
        raise ConversionError(f"{label} DICOM column direction is not unit length")
    if not math.isclose(
        float(np.dot(dicom_row_direction, dicom_column_direction)),
        0.0,
        abs_tol=ORIENTATION_TOLERANCE,
    ):
        raise ConversionError(f"{label} DICOM row/column directions are not orthogonal")
    for header in headers:
        if not np.allclose(
            header.image_orientation_patient,
            reference_iop,
            rtol=0.0,
            atol=ORIENTATION_TOLERANCE,
        ):
            raise ConversionError(f"{label} orientation varies across instances")

    reference_pixel_spacing = headers[0].pixel_spacing
    if np.any(reference_pixel_spacing <= 0):
        raise ConversionError(f"{label} PixelSpacing must be positive")
    for header in headers:
        if not np.allclose(
            header.pixel_spacing,
            reference_pixel_spacing,
            rtol=ORIENTATION_TOLERANCE,
            atol=ORIENTATION_TOLERANCE,
        ):
            raise ConversionError(f"{label} PixelSpacing varies across instances")

    thicknesses = np.asarray(
        [header.slice_thickness for header in headers], dtype=np.float64
    )
    if np.any(thicknesses <= 0) or not np.allclose(
        thicknesses,
        thicknesses[0],
        rtol=SPACING_RELATIVE_TOLERANCE,
        atol=POSITION_TOLERANCE_MM,
    ):
        raise ConversionError(f"{label} SliceThickness is invalid or inconsistent")

    slice_normal = np.cross(dicom_row_direction, dicom_column_direction)
    slice_normal /= np.linalg.norm(slice_normal)
    headers.sort(
        key=lambda header: float(
            np.dot(header.image_position_patient, slice_normal)
        )
    )
    projected_positions = np.asarray(
        [
            np.dot(header.image_position_patient, slice_normal)
            for header in headers
        ],
        dtype=np.float64,
    )
    projected_differences = np.diff(projected_positions)
    if np.any(projected_differences <= POSITION_TOLERANCE_MM):
        raise ConversionError(f"{label} contains duplicate physical slice positions")
    measured_spacing = float(np.median(projected_differences))
    spacing_tolerance = max(
        POSITION_TOLERANCE_MM,
        measured_spacing * SPACING_RELATIVE_TOLERANCE,
    )
    irregular = np.abs(projected_differences - measured_spacing) > spacing_tolerance
    if np.any(irregular):
        largest_gap = float(projected_differences.max())
        if largest_gap > 1.5 * measured_spacing:
            raise ConversionError(
                f"{label} appears to contain missing slices; largest gap is "
                f"{largest_gap:.12g} mm"
            )
        raise ConversionError(f"{label} has irregular inter-slice spacing")
    if not math.isclose(
        float(thicknesses[0]),
        measured_spacing,
        rel_tol=SPACING_RELATIVE_TOLERANCE,
        abs_tol=POSITION_TOLERANCE_MM,
    ):
        raise ConversionError(
            f"{label} SliceThickness does not match measured slice spacing"
        )

    first_position = headers[0].image_position_patient
    last_position = headers[-1].image_position_patient
    slice_step = (last_position - first_position) / (len(headers) - 1)
    in_plane_drifts = []
    position_errors = []
    for index, header in enumerate(headers):
        offset = header.image_position_patient - first_position
        projected_offset = slice_normal * np.dot(offset, slice_normal)
        in_plane_drifts.append(float(np.linalg.norm(offset - projected_offset)))
        expected_position = first_position + slice_step * index
        position_errors.append(
            float(np.linalg.norm(header.image_position_patient - expected_position))
        )
    maximum_in_plane_drift = max(in_plane_drifts)
    maximum_position_error = max(position_errors)
    if maximum_in_plane_drift > POSITION_TOLERANCE_MM:
        raise ConversionError(f"{label} has in-plane slice-origin drift")
    if maximum_position_error > spacing_tolerance:
        raise ConversionError(
            f"{label} slice positions do not follow one regular physical grid"
        )

    # DICOM IOP first triplet follows increasing pixel column; its second
    # triplet follows increasing pixel row. PixelSpacing is [row, column].
    lps_affine = np.eye(4, dtype=np.float64)
    lps_affine[:3, 0] = dicom_column_direction * reference_pixel_spacing[0]
    lps_affine[:3, 1] = dicom_row_direction * reference_pixel_spacing[1]
    lps_affine[:3, 2] = slice_step
    lps_affine[:3, 3] = first_position

    origin_from_affine = (lps_affine @ np.asarray([0, 0, 0, 1]))[:3]
    row_increment = (
        lps_affine @ np.asarray([1, 0, 0, 1])
    )[:3] - origin_from_affine
    column_increment = (
        lps_affine @ np.asarray([0, 1, 0, 1])
    )[:3] - origin_from_affine
    row_expected = dicom_column_direction * reference_pixel_spacing[0]
    column_expected = dicom_row_direction * reference_pixel_spacing[1]
    slice_positions_match = all(
        np.allclose(
            (lps_affine @ np.asarray([0, 0, index, 1]))[:3],
            header.image_position_patient,
            rtol=0.0,
            atol=spacing_tolerance,
        )
        for index, header in enumerate(headers)
    )
    semantic_checks = {
        "voxel_0_0_0_matches_first_image_position_patient": bool(
            np.allclose(
                origin_from_affine,
                first_position,
                rtol=0.0,
                atol=POSITION_TOLERANCE_MM,
            )
        ),
        "row_index_increment_matches_iop_and_pixel_spacing": bool(
            np.allclose(
                row_increment,
                row_expected,
                rtol=0.0,
                atol=ORIENTATION_TOLERANCE,
            )
        ),
        "column_index_increment_matches_iop_and_pixel_spacing": bool(
            np.allclose(
                column_increment,
                column_expected,
                rtol=0.0,
                atol=ORIENTATION_TOLERANCE,
            )
        ),
        "slice_index_increments_match_source_image_positions": slice_positions_match,
    }
    if not all(semantic_checks.values()):
        failed = [name for name, passed in semantic_checks.items() if not passed]
        raise ConversionError(
            f"{label} DICOM row/column affine semantics failed: {', '.join(failed)}"
        )

    ras_affine = LPS_TO_RAS @ lps_affine
    shape = (rows, columns, len(headers))
    if abs(float(np.linalg.det(ras_affine[:3, :3]))) <= 1e-12:
        raise ConversionError(f"{label} affine is singular")
    orientation_codes = tuple(str(code) for code in nib.aff2axcodes(ras_affine))
    bounds = _physical_bounds(shape, ras_affine)
    return Geometry(
        headers=tuple(headers),
        original_array_shape=shape,
        voxel_spacing=(
            float(reference_pixel_spacing[0]),
            float(reference_pixel_spacing[1]),
            float(np.linalg.norm(slice_step)),
        ),
        original_lps_affine=lps_affine,
        original_ras_affine=ras_affine,
        original_orientation_codes=orientation_codes,
        physical_bounds_ras_mm=bounds,
        maximum_slice_position_error_mm=maximum_position_error,
        maximum_in_plane_drift_mm=maximum_in_plane_drift,
        measured_slice_spacing_range_mm=(
            float(projected_differences.min()),
            float(projected_differences.max()),
        ),
        affine_semantics_checks=semantic_checks,
    )


def _select_series(
    result: inventory.InventoryResult,
) -> tuple[
    tuple[str, inventory.SeriesInventory],
    tuple[str, inventory.SeriesInventory],
]:
    pet_matches = []
    ct_matches = []
    for study_uid, series_by_uid in result.studies.items():
        for series in series_by_uid.values():
            description = str(series.metadata.get("SeriesDescription") or "")
            modality = str(series.metadata.get("Modality") or "")
            if description == PET_DESCRIPTION and modality == "PT":
                details = series.pet_details
                units = details.observations["Units"] if details else None
                if (
                    units is not None
                    and units.state == "CONSTANT"
                    and str(units.first_value).upper() == "BQML"
                ):
                    pet_matches.append((study_uid, series))
            if description == CT_DESCRIPTION and modality == "CT":
                ct_matches.append((study_uid, series))
    if len(pet_matches) != 1:
        raise ConversionError(
            f"expected exactly one {PET_DESCRIPTION!r} PT/BQML series; found {len(pet_matches)}"
        )
    if len(ct_matches) != 1:
        raise ConversionError(
            f"expected exactly one {CT_DESCRIPTION!r} CT series; found {len(ct_matches)}"
        )
    if pet_matches[0][0] != ct_matches[0][0]:
        raise ConversionError("selected PET and CT series belong to different studies")
    return pet_matches[0], ct_matches[0]


def _pet_factor_and_mapping(
    result: inventory.InventoryResult,
    study_uid: str,
    series: inventory.SeriesInventory,
) -> tuple[inventory.MetadataSUVFactor, inventory.RWVMapping, dict[str, float | str]]:
    if series.pet_details is None:
        raise ConversionError("PET metadata details are unavailable")
    factor = inventory._metadata_suvbw_factor(series.pet_details)
    if factor.value is None:
        raise ConversionError(
            "PET metadata-derived SUVbw factor cannot be calculated: "
            f"{factor.reason}"
        )
    if not inventory._has_decy_on_all_instances(
        series.pet_details.observations["CorrectedImage"]
    ):
        raise ConversionError(
            "PET CorrectedImage does not consistently contain the DECY flag"
        )
    mappings = inventory._rwv_mappings_for_pt_series(
        result, study_uid, series.series_instance_uid
    )
    mapping = inventory._suvbw_mapping(mappings)
    if mapping is None:
        raise ConversionError("linked SUVbw RWV mapping is unavailable")
    rwv_slope = inventory._mapping_number(mapping.slope)
    if rwv_slope is None:
        raise ConversionError("linked SUVbw RWV slope is unavailable")
    absolute_difference = abs(factor.value - rwv_slope)
    relative_difference = absolute_difference / abs(factor.value)
    percentage_difference = relative_difference * 100.0
    if percentage_difference > FACTOR_MATCH_PERCENT_TOLERANCE:
        raise ConversionError(
            "metadata-derived BQML-to-SUVbw factor does not match linked RWV "
            f"factor within {FACTOR_MATCH_PERCENT_TOLERANCE}%"
        )
    return factor, mapping, {
        "metadata_derived_bqml_to_suvbw_factor": factor.value,
        "rwv_suvbw_factor": rwv_slope,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "percentage_difference": percentage_difference,
        "result": "MATCH",
        "tolerance_percent": FACTOR_MATCH_PERCENT_TOLERANCE,
    }


def _build_volume(
    geometry: Geometry,
    *,
    kind: str,
    pet_factor: float | None = None,
    rwv_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, VolumeStatistics]:
    volume = np.empty(geometry.original_array_shape, dtype=np.float32)
    finite_count = 0
    non_finite_count = 0
    value_sum = 0.0
    value_min = math.inf
    value_max = -math.inf
    outside_rwv_count = 0 if rwv_range is not None else None
    outside_rwv_affected_slices = 0 if rwv_range is not None else None
    outside_rwv_minimum_index: np.ndarray | None = None
    outside_rwv_maximum_index: np.ndarray | None = None

    for index, header in enumerate(geometry.headers):
        try:
            dataset = pydicom.dcmread(header.path, force=False)
            pixels = np.asarray(dataset.pixel_array)
        except Exception as error:
            raise ConversionError(
                f"{kind} pixel decoding failed for SOPInstanceUID "
                f"{header.sop_instance_uid}; TransferSyntaxUID "
                f"{header.transfer_syntax_uid}; {_safe_error(error, header.path)}"
            ) from error
        if pixels.shape != geometry.original_array_shape[:2]:
            raise ConversionError(
                f"{kind} decoded slice shape {pixels.shape} does not match validated geometry"
            )
        if str(dataset.get("SOPInstanceUID", "")) != header.sop_instance_uid:
            raise ConversionError(f"{kind} source instance changed between validation and decoding")

        transformed = (
            pixels.astype(np.float64) * header.rescale_slope
            + header.rescale_intercept
        )
        if kind == "PET":
            if pet_factor is None:
                raise ConversionError("PET factor is unavailable")
            transformed *= pet_factor
            if rwv_range is not None and outside_rwv_count is not None:
                first_mapped, last_mapped = rwv_range
                outside_mask = (pixels < first_mapped) | (pixels > last_mapped)
                slice_outside_count = int(np.count_nonzero(outside_mask))
                outside_rwv_count += slice_outside_count
                if slice_outside_count:
                    if outside_rwv_affected_slices is None:
                        raise AssertionError("RWV affected-slice counter is unavailable")
                    outside_rwv_affected_slices += 1
                    row_column_indices = np.argwhere(outside_mask)
                    slice_minimum = np.asarray(
                        [
                            row_column_indices[:, 0].min(),
                            row_column_indices[:, 1].min(),
                            index,
                        ],
                        dtype=np.int64,
                    )
                    slice_maximum = np.asarray(
                        [
                            row_column_indices[:, 0].max(),
                            row_column_indices[:, 1].max(),
                            index,
                        ],
                        dtype=np.int64,
                    )
                    outside_rwv_minimum_index = (
                        slice_minimum
                        if outside_rwv_minimum_index is None
                        else np.minimum(outside_rwv_minimum_index, slice_minimum)
                    )
                    outside_rwv_maximum_index = (
                        slice_maximum
                        if outside_rwv_maximum_index is None
                        else np.maximum(outside_rwv_maximum_index, slice_maximum)
                    )
        output_slice = transformed.astype(np.float32)
        volume[:, :, index] = output_slice
        finite = np.isfinite(output_slice)
        slice_finite_count = int(np.count_nonzero(finite))
        finite_count += slice_finite_count
        non_finite_count += int(output_slice.size - slice_finite_count)
        if slice_finite_count:
            finite_values = output_slice[finite]
            value_sum += float(finite_values.sum(dtype=np.float64))
            value_min = min(value_min, float(finite_values.min()))
            value_max = max(value_max, float(finite_values.max()))

    if non_finite_count or not finite_count:
        raise ConversionError(f"{kind} derived volume contains non-finite voxels")
    outside_voxel_bounds = None
    outside_physical_bounds = None
    if outside_rwv_minimum_index is not None and outside_rwv_maximum_index is not None:
        outside_voxel_bounds = (
            tuple(int(value) for value in outside_rwv_minimum_index),
            tuple(int(value) for value in outside_rwv_maximum_index),
        )
        outside_shape = outside_rwv_maximum_index - outside_rwv_minimum_index + 1
        translated_affine = geometry.original_ras_affine.copy()
        translated_affine[:3, 3] = (
            geometry.original_ras_affine
            @ np.append(outside_rwv_minimum_index, 1.0)
        )[:3]
        outside_physical_bounds = _physical_bounds(outside_shape, translated_affine)

    return volume, VolumeStatistics(
        minimum=value_min,
        maximum=value_max,
        mean=value_sum / finite_count,
        finite_voxels=finite_count,
        non_finite_voxels=non_finite_count,
        out_of_rwv_range_voxels=outside_rwv_count,
        out_of_rwv_range_affected_slices=outside_rwv_affected_slices,
        out_of_rwv_range_original_voxel_bounds=outside_voxel_bounds,
        out_of_rwv_range_physical_bounds_ras_mm=outside_physical_bounds,
    )


def _canonicalize(
    volume: np.ndarray, geometry: Geometry, kind: str
) -> tuple[nib.Nifti1Image, CanonicalVolume]:
    original_image = nib.Nifti1Image(volume, geometry.original_ras_affine)
    original_orientation = nib.orientations.io_orientation(
        geometry.original_ras_affine
    )
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(
        original_orientation, target_orientation
    )
    canonical_image = nib.as_closest_canonical(original_image)
    expected_affine = geometry.original_ras_affine @ nib.orientations.inv_ornt_aff(
        transform, geometry.original_array_shape
    )
    if not np.allclose(
        canonical_image.affine, expected_affine, rtol=0.0, atol=1e-6
    ):
        raise ConversionError(
            f"{kind} canonicalization was not a pure permutation/flip"
        )
    if int(np.prod(canonical_image.shape)) != int(np.prod(volume.shape)):
        raise ConversionError(f"{kind} canonicalization changed voxel count")

    canonical_image.set_data_dtype(np.float32)
    canonical_image.header.set_xyzt_units("mm")
    canonical_image.header["descrip"] = f"Research-derived {kind} volume"
    canonical_image.set_qform(canonical_image.affine, code=1)
    canonical_image.set_sform(canonical_image.affine, code=1)
    canonical_codes = tuple(
        str(code) for code in nib.aff2axcodes(canonical_image.affine)
    )
    if canonical_codes != ("R", "A", "S"):
        raise ConversionError(f"{kind} canonical orientation is not RAS")

    canonical_bounds = _physical_bounds(canonical_image.shape, canonical_image.affine)
    if not np.allclose(
        np.asarray(canonical_bounds),
        np.asarray(geometry.physical_bounds_ras_mm),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ConversionError(f"{kind} canonicalization changed physical bounds")
    permutation = transform[:, 0].astype(int)
    return canonical_image, CanonicalVolume(
        output_array_shape=tuple(int(value) for value in canonical_image.shape),
        canonical_ras_affine=np.asarray(canonical_image.affine, dtype=np.float64),
        canonical_orientation_codes=canonical_codes,
        axes_permuted=not np.array_equal(permutation, np.arange(3)),
        axes_flipped=bool(np.any(transform[:, 1] == -1)),
        orientation_transform=transform,
        physical_bounds_ras_mm=canonical_bounds,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_written_nifti(
    path: Path,
    canonical: CanonicalVolume,
    statistics: VolumeStatistics,
) -> dict[str, bool]:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    finite = np.isfinite(data)
    finite_count = int(np.count_nonzero(finite))
    non_finite_count = int(data.size - finite_count)
    if finite_count == data.size:
        written_minimum = float(data.min())
        written_maximum = float(data.max())
        written_mean = float(data.sum(dtype=np.float64) / finite_count)
    elif finite_count:
        finite_values = data[finite]
        written_minimum = float(finite_values.min())
        written_maximum = float(finite_values.max())
        written_mean = float(finite_values.sum(dtype=np.float64) / finite_count)
    else:
        written_minimum = written_maximum = written_mean = math.nan
    checks = {
        "shape_matches": tuple(image.shape) == canonical.output_array_shape,
        "dtype_is_float32": np.dtype(image.get_data_dtype()) == np.dtype(np.float32),
        "affine_matches": bool(
            np.allclose(
                image.affine,
                canonical.canonical_ras_affine,
                rtol=0.0,
                atol=1e-5,
            )
        ),
        "orientation_is_ras": tuple(nib.aff2axcodes(image.affine))
        == ("R", "A", "S"),
        "qform_code_is_set": int(image.header["qform_code"]) > 0,
        "sform_code_is_set": int(image.header["sform_code"]) > 0,
        "voxel_spacing_matches": bool(
            np.allclose(
                image.header.get_zooms()[:3],
                np.linalg.norm(canonical.canonical_ras_affine[:3, :3], axis=0),
                rtol=1e-5,
                atol=1e-5,
            )
        ),
        "physical_bounds_match": bool(
            np.allclose(
                np.asarray(_physical_bounds(image.shape, image.affine)),
                np.asarray(canonical.physical_bounds_ras_mm),
                rtol=0.0,
                atol=1e-4,
            )
        ),
        "voxel_count_matches": int(data.size)
        == statistics.finite_voxels + statistics.non_finite_voxels,
        "finite_voxel_count_matches": finite_count == statistics.finite_voxels,
        "non_finite_voxel_count_matches": non_finite_count
        == statistics.non_finite_voxels,
        "minimum_matches": written_minimum == statistics.minimum,
        "maximum_matches": written_maximum == statistics.maximum,
        "mean_matches": math.isclose(
            written_mean, statistics.mean, rel_tol=1e-12, abs_tol=1e-12
        ),
        "statistics_are_finite": all(
            math.isfinite(value)
            for value in (statistics.minimum, statistics.maximum, statistics.mean)
        ),
        "all_voxels_are_finite": statistics.non_finite_voxels == 0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ConversionError(
            f"post-write NIfTI verification failed: {', '.join(failed)}"
        )
    return checks


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _json_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> dict[str, list[float]]:
    return {"minimum": list(bounds[0]), "maximum": list(bounds[1])}


def _geometry_provenance(
    geometry: Geometry, canonical: CanonicalVolume
) -> dict[str, Any]:
    return {
        "array_index_order": ["row", "column", "slice"],
        "original_array_shape": list(geometry.original_array_shape),
        "original_lps_affine": _json_matrix(geometry.original_lps_affine),
        "original_ras_affine": _json_matrix(geometry.original_ras_affine),
        "original_orientation_codes": list(geometry.original_orientation_codes),
        "dicom_image_orientation_patient": list(
            geometry.headers[0].image_orientation_patient
        ),
        "output_array_shape": list(canonical.output_array_shape),
        "canonical_ras_affine": _json_matrix(canonical.canonical_ras_affine),
        "canonical_orientation_codes": list(canonical.canonical_orientation_codes),
        "canonicalization_axes_permuted": canonical.axes_permuted,
        "canonicalization_axes_flipped": canonical.axes_flipped,
        "canonicalization_orientation_transform": _json_matrix(
            canonical.orientation_transform
        ),
        "canonicalization_method": "nibabel.as_closest_canonical; permutation/flipping only; no interpolation or resampling",
        "voxel_spacing_mm": list(
            np.linalg.norm(canonical.canonical_ras_affine[:3, :3], axis=0)
        ),
        "physical_bounds_ras_mm_voxel_centers": _json_bounds(
            canonical.physical_bounds_ras_mm
        ),
        "measured_slice_spacing_range_mm": list(
            geometry.measured_slice_spacing_range_mm
        ),
        "maximum_slice_position_error_mm": geometry.maximum_slice_position_error_mm,
        "maximum_in_plane_origin_drift_mm": geometry.maximum_in_plane_drift_mm,
        "dicom_affine_semantics_checks": geometry.affine_semantics_checks,
    }


def _provenance(
    *,
    kind: str,
    study_uid: str,
    series: inventory.SeriesInventory,
    written: WrittenVolume,
    factor: inventory.MetadataSUVFactor | None = None,
    mapping: inventory.RWVMapping | None = None,
    factor_agreement: dict[str, float | str] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "artifact": written.path.name.replace(".tmp", ""),
        "research_use_only": True,
        "source": {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series.series_instance_uid,
            "SeriesDescription": str(series.metadata.get("SeriesDescription") or ""),
            "Modality": str(series.metadata.get("Modality") or ""),
            "SOPClassUID": str(series.metadata.get("SOPClassUID") or ""),
            "NumberOfInstances": len(written.geometry.headers),
            "SourceSOPInstanceUIDCount": len(
                {header.sop_instance_uid for header in written.geometry.headers}
            ),
        },
        "geometry": _geometry_provenance(written.geometry, written.canonical),
        "statistics": {
            "minimum": written.statistics.minimum,
            "maximum": written.statistics.maximum,
            "mean": written.statistics.mean,
            "finite_voxel_count": written.statistics.finite_voxels,
            "non_finite_voxel_count": written.statistics.non_finite_voxels,
        },
        "post_write_verification": written.post_write_checks,
        "output": {
            "sha256": written.sha256,
            "size_bytes": written.size_bytes,
        },
        "software": {
            "petct-research": importlib.metadata.version("petct-research"),
            "numpy": np.__version__,
            "pydicom": pydicom.__version__,
            "nibabel": nib.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if kind == "PET":
        if factor is None or mapping is None or factor_agreement is None:
            raise ConversionError("PET provenance inputs are incomplete")
        provenance["source"]["Units"] = "BQML"
        details = series.pet_details
        if details is None:
            raise ConversionError("PET details are unavailable for provenance")
        image_suv_type = details.observations["SUVType"].first_value
        corrected_image = details.observations["CorrectedImage"].first_value
        provenance["source"]["SUVType"] = mapping.suv_type
        provenance["source"]["DICOMPETImageSUVType"] = image_suv_type
        provenance["source"]["DecayCorrection"] = details.observations[
            "DecayCorrection"
        ].first_value
        provenance["source"]["CorrectedImage"] = (
            list(corrected_image)
            if isinstance(corrected_image, tuple)
            else corrected_image
        )
        provenance["transformation"] = {
            "name": "CandidateSUVbwTransformation",
            "status": "candidate quantitative transformation; not clinically validated",
            "bqml_formula": "StoredValue * InstanceRescaleSlope + InstanceRescaleIntercept",
            "suvbw_formula": "BQML * MetadataDerivedBQMLToSUVbwFactor",
            "metadata_derived_bqml_to_suvbw_factor": factor.value,
            "decay_timing_basis": factor.basis,
            "injection_to_reference_seconds": factor.elapsed_seconds,
            "decay_corrected_injected_activity_bq": factor.decay_corrected_dose_bq,
            "rwv_suvbw_factor": inventory._mapping_number(mapping.slope),
            "factor_agreement": factor_agreement,
            "rwv_declared_stored_value_range": [
                inventory._mapping_number(mapping.first_value_mapped),
                inventory._mapping_number(mapping.last_value_mapped),
            ],
            "voxels_outside_rwv_range_preserved": written.statistics.out_of_rwv_range_voxels,
            "out_of_rwv_range_location": {
                "affected_source_slices": written.statistics.out_of_rwv_range_affected_slices,
                "original_array_voxel_index_bounds": (
                    {
                        "minimum": list(
                            written.statistics.out_of_rwv_range_original_voxel_bounds[0]
                        ),
                        "maximum": list(
                            written.statistics.out_of_rwv_range_original_voxel_bounds[1]
                        ),
                    }
                    if written.statistics.out_of_rwv_range_original_voxel_bounds
                    else None
                ),
                "physical_bounds_ras_mm_voxel_centers": (
                    _json_bounds(
                        written.statistics.out_of_rwv_range_physical_bounds_ras_mm
                    )
                    if written.statistics.out_of_rwv_range_physical_bounds_ras_mm
                    else None
                ),
            },
            "out_of_range_policy": "preserved through candidate BQML-to-SUVbw transformation; not clipped, zeroed, or replaced",
        }
    else:
        provenance["transformation"] = {
            "name": "CTStoredValueToHU",
            "formula": "StoredValue * InstanceRescaleSlope + InstanceRescaleIntercept",
        }
    return provenance


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _bounds_overlap(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    first_min, first_max = np.asarray(first[0]), np.asarray(first[1])
    second_min, second_max = np.asarray(second[0]), np.asarray(second[1])
    return bool(np.all(np.minimum(first_max, second_max) >= np.maximum(first_min, second_min)))


def _output_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "output" / "nifti"


def convert_study(source: Path, *, overwrite: bool = False) -> ConversionReport:
    result = inventory.inventory_directory(
        source,
        include_pet_details=True,
        retain_source_paths=True,
    )
    (study_uid, pet_series), (_ct_study_uid, ct_series) = _select_series(result)
    pet_factor, rwv_mapping, factor_agreement = _pet_factor_and_mapping(
        result, study_uid, pet_series
    )
    rwv_range = inventory._mapping_range(rwv_mapping)
    if rwv_range is None:
        raise ConversionError("linked SUVbw RWV mapped range is unavailable")

    pet_geometry = _validate_geometry(_read_slice_headers(pet_series), "PET")
    ct_geometry = _validate_geometry(_read_slice_headers(ct_series), "CT")
    if not _bounds_overlap(
        pet_geometry.physical_bounds_ras_mm,
        ct_geometry.physical_bounds_ras_mm,
    ):
        raise ConversionError("PET and CT physical bounds do not overlap")

    output_directory = _output_directory()
    output_directory.mkdir(parents=True, exist_ok=True)
    final_paths = {
        name: output_directory / filename
        for name, filename in OUTPUT_FILENAMES.items()
    }
    existing = [path.name for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise ConversionError(
            "derived outputs already exist; use --overwrite to replace only the "
            "known output files"
        )
    temporary_paths = {
        "pet_nifti": output_directory / ".PET_SUVbw.tmp.nii.gz",
        "ct_nifti": output_directory / ".CT_WB_CECT.tmp.nii.gz",
        "pet_provenance": output_directory / ".pet_provenance.tmp.json",
        "ct_provenance": output_directory / ".ct_provenance.tmp.json",
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    try:
        pet_volume, pet_statistics = _build_volume(
            pet_geometry,
            kind="PET",
            pet_factor=pet_factor.value,
            rwv_range=rwv_range,
        )
        pet_image, pet_canonical = _canonicalize(pet_volume, pet_geometry, "PET")
        nib.save(pet_image, temporary_paths["pet_nifti"])
        del pet_image, pet_volume
        gc.collect()
        pet_checks = _verify_written_nifti(
            temporary_paths["pet_nifti"], pet_canonical, pet_statistics
        )
        pet_written = WrittenVolume(
            kind="PET",
            path=final_paths["pet_nifti"],
            geometry=pet_geometry,
            canonical=pet_canonical,
            statistics=pet_statistics,
            sha256=_sha256(temporary_paths["pet_nifti"]),
            size_bytes=temporary_paths["pet_nifti"].stat().st_size,
            post_write_checks=pet_checks,
        )
        ct_volume, ct_statistics = _build_volume(ct_geometry, kind="CT")
        ct_image, ct_canonical = _canonicalize(ct_volume, ct_geometry, "CT")
        nib.save(ct_image, temporary_paths["ct_nifti"])
        del ct_image, ct_volume
        gc.collect()
        ct_checks = _verify_written_nifti(
            temporary_paths["ct_nifti"], ct_canonical, ct_statistics
        )
        ct_written = WrittenVolume(
            kind="CT",
            path=final_paths["ct_nifti"],
            geometry=ct_geometry,
            canonical=ct_canonical,
            statistics=ct_statistics,
            sha256=_sha256(temporary_paths["ct_nifti"]),
            size_bytes=temporary_paths["ct_nifti"].stat().st_size,
            post_write_checks=ct_checks,
        )

        _write_json(
            temporary_paths["pet_provenance"],
            _provenance(
                kind="PET",
                study_uid=study_uid,
                series=pet_series,
                written=pet_written,
                factor=pet_factor,
                mapping=rwv_mapping,
                factor_agreement=factor_agreement,
            ),
        )
        _write_json(
            temporary_paths["ct_provenance"],
            _provenance(
                kind="CT",
                study_uid=study_uid,
                series=ct_series,
                written=ct_written,
            ),
        )
        for name in OUTPUT_FILENAMES:
            os.replace(temporary_paths[name], final_paths[name])
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    return ConversionReport(
        pet=pet_written,
        ct=ct_written,
        physical_bounds_overlap=True,
    )


def _format_vector(values: Sequence[float]) -> str:
    return " x ".join(f"{value:.12g}" for value in values)


def _format_written_volume(volume: WrittenVolume) -> list[str]:
    label = "SUV" if volume.kind == "PET" else "HU"
    lines = [
        f"{volume.kind}:",
        f"  Shape: {volume.canonical.output_array_shape}",
        "  VoxelSpacingMm: "
        f"{_format_vector(np.linalg.norm(volume.canonical.canonical_ras_affine[:3, :3], axis=0))}",
        f"  Orientation: {''.join(volume.canonical.canonical_orientation_codes)}",
        "  PhysicalBoundsRASMm: "
        f"minimum={volume.canonical.physical_bounds_ras_mm[0]}, "
        f"maximum={volume.canonical.physical_bounds_ras_mm[1]}",
        f"  {label}Minimum: {volume.statistics.minimum:.12g}",
        f"  {label}Maximum: {volume.statistics.maximum:.12g}",
        f"  {label}Mean: {volume.statistics.mean:.12g}",
        f"  FiniteVoxels: {volume.statistics.finite_voxels}",
        f"  NonFiniteVoxels: {volume.statistics.non_finite_voxels}",
    ]
    if volume.statistics.out_of_rwv_range_voxels is not None:
        lines.append(
            "  OutOfRWVRangeVoxels: "
            f"{volume.statistics.out_of_rwv_range_voxels}"
        )
    lines.extend(
        [
            f"  SHA256: {volume.sha256}",
            f"  FileSizeBytes: {volume.size_bytes}",
            f"  CanonicalizationAxesPermuted: {volume.canonical.axes_permuted}",
            f"  CanonicalizationAxesFlipped: {volume.canonical.axes_flipped}",
            "  PostWriteVerification: PASS",
        ]
    )
    return lines


def format_report(report: ConversionReport) -> str:
    lines = _format_written_volume(report.pet)
    pet_provenance = json.loads(
        (report.pet.path.parent / OUTPUT_FILENAMES["pet_provenance"]).read_text()
    )
    agreement = pet_provenance["transformation"]["factor_agreement"]
    lines.extend(
        [
            "  MetadataDerivedBQMLToSUVbwFactor: "
            f"{agreement['metadata_derived_bqml_to_suvbw_factor']:.12g}",
            f"  RWVSUVbwFactor: {agreement['rwv_suvbw_factor']:.12g}",
            f"  FactorAgreement: {agreement['result']}",
        ]
    )
    lines.extend(_format_written_volume(report.ct))
    lines.extend(
        [
            "Geometry:",
            "  PETCTPhysicalBoundsOverlap: "
            f"{'PASS' if report.physical_bounds_overlap else 'FAIL'}",
            "  AllPostWriteVerification: PASS",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create verified derived PET SUVbw and WB CECT NIfTI volumes."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="source DICOM directory located outside this repository",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the four known derived outputs if they already exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = convert_study(args.source, overwrite=args.overwrite)
    except ConversionError as error:
        parser.exit(1, f"conversion failed: {error}\n")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
