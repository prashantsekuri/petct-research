"""Prepare paired CT/PET NIfTI inputs for GLOW-FDG without registration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PET = PROJECT_ROOT / "output/nifti/PET_SUVbw.nii.gz"
SOURCE_CT = PROJECT_ROOT / "output/nifti/CT_WB_CECT.nii.gz"
OUTPUT_DIRECTORY = PROJECT_ROOT / "output/segmentation/glow_fdg/input"
OUTPUT_CT = OUTPUT_DIRECTORY / "study_0000.nii.gz"
OUTPUT_PET = OUTPUT_DIRECTORY / "study_0001.nii.gz"
OUTPUT_PROVENANCE = OUTPUT_DIRECTORY / "preprocessing_provenance.json"
OUTPUT_PATHS = (OUTPUT_CT, OUTPUT_PET, OUTPUT_PROVENANCE)

CT_OUTSIDE_FOV_HU = -1024.0
AFFINE_TOLERANCE = 1e-6
PET_THRESHOLDS = (0.0, 0.1, 0.5)


class PreparationError(RuntimeError):
    """Raised when paired input preparation cannot be completed safely."""


@dataclass(frozen=True)
class ImageSummary:
    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]
    orientation: tuple[str, str, str]
    physical_voxel_center_bounds_mm: (
        tuple[tuple[float, float, float], tuple[float, float, float]]
    )


@dataclass(frozen=True)
class ArrayStatistics:
    minimum: float
    maximum: float
    mean: float
    finite_voxels: int
    non_finite_voxels: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_bounds(
    shape: Sequence[int], affine: np.ndarray
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    corners = []
    for index_0 in (0, shape[0] - 1):
        for index_1 in (0, shape[1] - 1):
            for index_2 in (0, shape[2] - 1):
                point = affine @ np.asarray(
                    [index_0, index_1, index_2, 1.0], dtype=np.float64
                )
                corners.append(point[:3])
    corner_array = np.asarray(corners, dtype=np.float64)
    return (
        tuple(float(value) for value in corner_array.min(axis=0)),
        tuple(float(value) for value in corner_array.max(axis=0)),
    )


def _load_image(path: Path, label: str) -> tuple[nib.spatialimages.SpatialImage, ImageSummary]:
    if not path.is_file():
        raise PreparationError(f"{label} input is missing at its expected output location")
    try:
        image = nib.load(path)
    except Exception as error:
        raise PreparationError(f"cannot load {label} NIfTI: {type(error).__name__}: {error}") from error
    if len(image.shape) != 3:
        raise PreparationError(f"{label} input must be three-dimensional")
    shape = tuple(int(value) for value in image.shape)
    if any(value <= 0 for value in shape):
        raise PreparationError(f"{label} input has an invalid shape")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise PreparationError(f"{label} affine is invalid or non-finite")
    if abs(float(np.linalg.det(affine[:3, :3]))) <= 1e-12:
        raise PreparationError(f"{label} affine is singular")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine))
    orientation = tuple(str(value) for value in nib.aff2axcodes(affine))
    return image, ImageSummary(
        shape=shape,
        affine=affine,
        spacing=spacing,
        orientation=orientation,
        physical_voxel_center_bounds_mm=_physical_bounds(shape, affine),
    )


def _array_statistics(data: np.ndarray) -> ArrayStatistics:
    finite = np.isfinite(data)
    finite_count = int(np.count_nonzero(finite))
    non_finite_count = int(data.size - finite_count)
    if not finite_count:
        raise PreparationError("image contains no finite voxels")
    finite_values = data if non_finite_count == 0 else data[finite]
    return ArrayStatistics(
        minimum=float(np.min(finite_values)),
        maximum=float(np.max(finite_values)),
        mean=float(np.sum(finite_values, dtype=np.float64) / finite_count),
        finite_voxels=finite_count,
        non_finite_voxels=non_finite_count,
    )


def _aabb_overlap(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[bool, tuple[tuple[float, float, float], tuple[float, float, float]] | None]:
    minimum = np.maximum(np.asarray(first[0]), np.asarray(second[0]))
    maximum = np.minimum(np.asarray(first[1]), np.asarray(second[1]))
    overlaps = bool(np.all(maximum >= minimum))
    if not overlaps:
        return False, None
    return True, (
        tuple(float(value) for value in minimum),
        tuple(float(value) for value in maximum),
    )


def _sample_points(
    pet: ImageSummary, ct: ImageSummary, target_to_source: np.ndarray
) -> list[dict[str, Any]]:
    points: list[tuple[str, tuple[int, int, int]]] = []
    for index_0 in (0, pet.shape[0] - 1):
        for index_1 in (0, pet.shape[1] - 1):
            for index_2 in (0, pet.shape[2] - 1):
                points.append(("corner", (index_0, index_1, index_2)))
    points.append(("center", tuple(value // 2 for value in pet.shape)))

    results = []
    ct_maximum = np.asarray(ct.shape, dtype=np.float64) - 1.0
    for kind, voxel in points:
        homogeneous_voxel = np.asarray([*voxel, 1.0], dtype=np.float64)
        world = pet.affine @ homogeneous_voxel
        source_voxel = target_to_source @ homogeneous_voxel
        inside = bool(
            np.all(source_voxel[:3] >= -AFFINE_TOLERANCE)
            and np.all(source_voxel[:3] <= ct_maximum + AFFINE_TOLERANCE)
        )
        results.append(
            {
                "kind": kind,
                "pet_voxel": list(voxel),
                "world_coordinate_mm": [float(value) for value in world[:3]],
                "source_ct_voxel": [float(value) for value in source_voxel[:3]],
                "inside_ct_fov": inside,
            }
        )
    return results


def _coverage_by_pet_centers(
    pet: ImageSummary, ct: ImageSummary, target_to_source: np.ndarray
) -> tuple[int, int]:
    """Count target voxel centers inside the CT voxel-center domain in z chunks."""
    index_0, index_1 = np.meshgrid(
        np.arange(pet.shape[0], dtype=np.float64),
        np.arange(pet.shape[1], dtype=np.float64),
        indexing="ij",
    )
    ct_maximum = np.asarray(ct.shape, dtype=np.float64) - 1.0
    covered = 0
    for index_2 in range(pet.shape[2]):
        coordinates = np.stack(
            (
                index_0,
                index_1,
                np.full(index_0.shape, index_2, dtype=np.float64),
            )
        ).reshape(3, -1)
        source = (
            target_to_source[:3, :3] @ coordinates
            + target_to_source[:3, 3, np.newaxis]
        )
        inside = np.all(source >= -AFFINE_TOLERANCE, axis=0) & np.all(
            source <= ct_maximum[:, np.newaxis] + AFFINE_TOLERANCE, axis=0
        )
        covered += int(np.count_nonzero(inside))
    return covered, int(np.prod(pet.shape, dtype=np.int64))


def _mask_extent(
    mask: np.ndarray, affine: np.ndarray
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    result: dict[str, Any] = {"voxel_count": count}
    if count == 0:
        return result
    occupied_0 = np.flatnonzero(np.any(mask, axis=(1, 2)))
    occupied_1 = np.flatnonzero(np.any(mask, axis=(0, 2)))
    occupied_2 = np.flatnonzero(np.any(mask, axis=(0, 1)))
    minimum = (
        int(occupied_0[0]),
        int(occupied_1[0]),
        int(occupied_2[0]),
    )
    maximum = (
        int(occupied_0[-1]),
        int(occupied_1[-1]),
        int(occupied_2[-1]),
    )
    extent_shape = tuple(maximum[axis] - minimum[axis] + 1 for axis in range(3))
    extent_affine = affine.copy()
    extent_affine[:3, 3] = (affine @ np.asarray([*minimum, 1.0]))[:3]
    bounds = _physical_bounds(extent_shape, extent_affine)
    result["physical_voxel_center_bounds_mm"] = _bounds_json(bounds)
    return result


def _alignment_diagnostic(
    pet_data: np.ndarray, ct_on_pet_grid: np.ndarray, affine: np.ndarray
) -> dict[str, Any]:
    pet_extents = {}
    for threshold in PET_THRESHOLDS:
        label = f"suvbw_gt_{threshold:g}"
        pet_extents[label] = _mask_extent(pet_data > threshold, affine)
    ct_extent = _mask_extent(ct_on_pet_grid > -500.0, affine)
    comparisons = {}
    if "physical_voxel_center_bounds_mm" in ct_extent:
        ct_bounds_json = ct_extent["physical_voxel_center_bounds_mm"]
        ct_bounds = (
            tuple(ct_bounds_json["minimum"]),
            tuple(ct_bounds_json["maximum"]),
        )
        ct_center = (
            np.asarray(ct_bounds[0], dtype=np.float64)
            + np.asarray(ct_bounds[1], dtype=np.float64)
        ) / 2.0
        for label, extent in pet_extents.items():
            if "physical_voxel_center_bounds_mm" not in extent:
                continue
            pet_bounds_json = extent["physical_voxel_center_bounds_mm"]
            pet_bounds = (
                tuple(pet_bounds_json["minimum"]),
                tuple(pet_bounds_json["maximum"]),
            )
            pet_center = (
                np.asarray(pet_bounds[0], dtype=np.float64)
                + np.asarray(pet_bounds[1], dtype=np.float64)
            ) / 2.0
            overlaps, overlap_bounds = _aabb_overlap(pet_bounds, ct_bounds)
            comparisons[label] = {
                "ct_extent_aabb_overlap": overlaps,
                "overlap_physical_voxel_center_bounds_mm": (
                    _bounds_json(overlap_bounds)
                    if overlap_bounds is not None
                    else None
                ),
                "bounding_box_center_distance_mm": float(
                    np.linalg.norm(pet_center - ct_center)
                ),
            }
    return {
        "description": "Descriptive threshold extents only; not an alignment pass/fail test.",
        "pet_extents": pet_extents,
        "ct_tissue_extent_hu_gt_minus_500": ct_extent,
        "pet_extent_vs_ct_tissue_extent": comparisons,
    }


def _bounds_json(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> dict[str, list[float]]:
    return {"minimum": list(bounds[0]), "maximum": list(bounds[1])}


def _summary_json(summary: ImageSummary) -> dict[str, Any]:
    return {
        "shape": list(summary.shape),
        "affine": summary.affine.tolist(),
        "voxel_spacing_mm": list(summary.spacing),
        "orientation_codes": list(summary.orientation),
        "physical_voxel_center_bounds_mm": _bounds_json(
            summary.physical_voxel_center_bounds_mm
        ),
    }


def _statistics_json(statistics: ArrayStatistics) -> dict[str, float | int]:
    return {
        "minimum": statistics.minimum,
        "maximum": statistics.maximum,
        "mean": statistics.mean,
        "finite_voxels": statistics.finite_voxels,
        "non_finite_voxels": statistics.non_finite_voxels,
    }


def _format_vector(values: Sequence[float], precision: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _format_bounds(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
) -> str:
    return f"min={_format_vector(bounds[0])}, max={_format_vector(bounds[1])}"


def _write_ct(
    data: np.ndarray,
    pet_image: nib.spatialimages.SpatialImage,
    temporary_path: Path,
) -> None:
    header = pet_image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_xyzt_units("mm")
    header["descrip"] = "CT HU resampled to PET grid for GLOW-FDG"
    image = nib.Nifti1Image(data, pet_image.affine, header=header)
    qform_code = int(pet_image.header["qform_code"])
    sform_code = int(pet_image.header["sform_code"])
    image.set_qform(pet_image.affine, code=qform_code or 1)
    image.set_sform(pet_image.affine, code=sform_code or 1)
    nib.save(image, temporary_path)


def _verify_outputs(
    source_pet: Path,
    output_pet: Path,
    output_ct: Path,
    pet: ImageSummary,
    expected_ct_statistics: ArrayStatistics,
    outside_count: int,
    target_to_source: np.ndarray,
    ct: ImageSummary,
) -> tuple[dict[str, bool], ArrayStatistics]:
    pet_output_image, pet_output = _load_image(output_pet, "copied PET")
    ct_output_image, ct_output = _load_image(output_ct, "resampled CT")
    ct_output_data = np.asanyarray(ct_output_image.dataobj, dtype=np.float32)
    ct_statistics = _array_statistics(ct_output_data)

    checks = {
        "pet_byte_identical": _sha256(source_pet) == _sha256(output_pet),
        "pet_shape_equal": pet_output.shape == pet.shape,
        "pet_affine_exactly_equal": bool(np.array_equal(pet_output.affine, pet.affine)),
        "pet_spacing_equal": bool(np.allclose(pet_output.spacing, pet.spacing, rtol=0.0, atol=0.0)),
        "pet_orientation_equal": pet_output.orientation == pet.orientation,
        "ct_shape_equals_pet": ct_output.shape == pet.shape,
        "ct_affine_exactly_equals_pet": bool(np.array_equal(ct_output.affine, pet.affine)),
        "ct_spacing_equals_pet": bool(np.allclose(ct_output.spacing, pet.spacing, rtol=0.0, atol=0.0)),
        "ct_orientation_equals_pet": ct_output.orientation == pet.orientation,
        "ct_finite": ct_statistics.non_finite_voxels == 0,
        "ct_statistics_preserved_on_write": bool(
            math.isclose(ct_statistics.minimum, expected_ct_statistics.minimum, abs_tol=1e-6)
            and math.isclose(ct_statistics.maximum, expected_ct_statistics.maximum, abs_tol=1e-6)
            and math.isclose(ct_statistics.mean, expected_ct_statistics.mean, rel_tol=1e-7, abs_tol=1e-6)
        ),
    }

    # Independently verify that every geometrically out-of-FOV output center has
    # the requested fill value without treating in-FOV -1024 HU as filled.
    if outside_count:
        index_0, index_1 = np.meshgrid(
            np.arange(pet.shape[0], dtype=np.float64),
            np.arange(pet.shape[1], dtype=np.float64),
            indexing="ij",
        )
        ct_maximum = np.asarray(ct.shape, dtype=np.float64) - 1.0
        outside_values_correct = True
        verified_outside = 0
        for index_2 in range(pet.shape[2]):
            coordinates = np.stack(
                (
                    index_0,
                    index_1,
                    np.full(index_0.shape, index_2, dtype=np.float64),
                )
            ).reshape(3, -1)
            source = (
                target_to_source[:3, :3] @ coordinates
                + target_to_source[:3, 3, np.newaxis]
            )
            inside = np.all(source >= -AFFINE_TOLERANCE, axis=0) & np.all(
                source <= ct_maximum[:, np.newaxis] + AFFINE_TOLERANCE, axis=0
            )
            outside = ~inside
            verified_outside += int(np.count_nonzero(outside))
            if np.any(ct_output_data[:, :, index_2].reshape(-1)[outside] != CT_OUTSIDE_FOV_HU):
                outside_values_correct = False
                break
        checks["outside_fov_count_reproduced"] = verified_outside == outside_count
        checks["outside_fov_voxels_have_fill_value"] = outside_values_correct
    else:
        checks["outside_fov_count_reproduced"] = True
        checks["outside_fov_voxels_have_fill_value"] = True

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise PreparationError("post-write verification failed: " + ", ".join(failed))
    del pet_output_image
    return checks, ct_statistics


def _provenance(
    *,
    pet: ImageSummary,
    ct: ImageSummary,
    source_pet_hash: str,
    source_ct_hash: str,
    output_pet_hash: str,
    output_ct_hash: str,
    pet_statistics: ArrayStatistics,
    ct_statistics: ArrayStatistics,
    target_to_source: np.ndarray,
    overlap: bool,
    overlap_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    covered_centers: int,
    total_centers: int,
    sample_points: list[dict[str, Any]],
    alignment: dict[str, Any],
    post_write_checks: dict[str, bool],
) -> dict[str, Any]:
    outside_count = total_centers - covered_centers
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Research-only paired input preparation for GLOW-FDG",
        "inputs": {
            "pet": {
                "sha256": source_pet_hash,
                **_summary_json(pet),
                "statistics_suvbw": _statistics_json(pet_statistics),
            },
            "ct": {"sha256": source_ct_hash, **_summary_json(ct)},
        },
        "target_grid": {
            "authoritative_modality": "PET",
            **_summary_json(pet),
        },
        "resampling": {
            "registration_performed": False,
            "ct_interpolation": "linear",
            "scipy_interpolation_order": 1,
            "prefilter": False,
            "outside_fov_fill_hu": CT_OUTSIDE_FOV_HU,
            "pet_resampled": False,
            "pet_normalized": False,
            "pet_clipped": False,
            "ct_normalized": False,
            "ct_clipped": False,
            "pet_grid_to_source_ct_voxel_transform": target_to_source.tolist(),
        },
        "geometry_checks": {
            "pet_ct_aabb_overlap": overlap,
            "overlap_physical_voxel_center_bounds_mm": (
                _bounds_json(overlap_bounds) if overlap_bounds is not None else None
            ),
            "pet_centers_inside_ct_voxel_center_fov": covered_centers,
            "pet_center_count": total_centers,
            "pet_center_coverage_percentage": covered_centers / total_centers * 100.0,
            "outside_ct_fov_fill_voxels": outside_count,
            "outside_ct_fov_fill_percentage": outside_count / total_centers * 100.0,
            "deterministic_sample_points": sample_points,
            "coarse_alignment_diagnostic": alignment,
        },
        "outputs": {
            "pet_channel_0001": {
                "sha256": output_pet_hash,
                "byte_identical_to_source": output_pet_hash == source_pet_hash,
                **_summary_json(pet),
                "statistics_suvbw": _statistics_json(pet_statistics),
            },
            "ct_channel_0000": {
                "sha256": output_ct_hash,
                **_summary_json(pet),
                "statistics_hu": _statistics_json(ct_statistics),
            },
        },
        "post_write_verification": post_write_checks,
        "software": {
            "petct_research": importlib.metadata.version("petct-research"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "scipy": importlib.metadata.version("scipy"),
        },
    }


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="ascii") as file:
        json.dump(document, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    os.replace(temporary, path)


def _print_report(
    *,
    pet: ImageSummary,
    ct: ImageSummary,
    pet_statistics: ArrayStatistics,
    ct_statistics: ArrayStatistics,
    source_pet_hash: str,
    output_pet_hash: str,
    output_ct_hash: str,
    covered_centers: int,
    total_centers: int,
    sample_points: list[dict[str, Any]],
    alignment: dict[str, Any],
    overlap: bool,
    overlap_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    post_write_checks: dict[str, bool],
) -> None:
    outside_count = total_centers - covered_centers
    print("PET PRESERVATION")
    print(f"  ByteIdentical: {'YES' if source_pet_hash == output_pet_hash else 'NO'}")
    print(f"  SourceSHA256: {source_pet_hash}")
    print(f"  OutputSHA256: {output_pet_hash}")
    print(
        "  SUVbwStats: "
        f"min={pet_statistics.minimum:.9g}, max={pet_statistics.maximum:.9g}, "
        f"mean={pet_statistics.mean:.9g}"
    )
    print(
        f"  FiniteVoxels: {pet_statistics.finite_voxels}; "
        f"NonFiniteVoxels: {pet_statistics.non_finite_voxels}"
    )

    print("\nCT RESAMPLING")
    print(f"  OutputShape: {pet.shape}")
    print(
        "  AffineMatchPET: "
        f"{'YES' if post_write_checks['ct_affine_exactly_equals_pet'] else 'NO'}"
    )
    print(f"  VoxelSpacingMm: {_format_vector(pet.spacing)}")
    print(f"  Orientation: {''.join(pet.orientation)}")
    print(
        "  HUStats: "
        f"min={ct_statistics.minimum:.9g}, max={ct_statistics.maximum:.9g}, "
        f"mean={ct_statistics.mean:.9g}"
    )
    print(
        f"  FiniteVoxels: {ct_statistics.finite_voxels}; "
        f"NonFiniteVoxels: {ct_statistics.non_finite_voxels}"
    )
    print(
        f"  OutsideCTFOVFill: {outside_count} / {total_centers} "
        f"({outside_count / total_centers * 100.0:.6f}%)"
    )
    print(f"  OutputSHA256: {output_ct_hash}")

    print("\nGEOMETRY")
    print(
        "  PET PhysicalVoxelCenterBoundsMm: "
        + _format_bounds(pet.physical_voxel_center_bounds_mm)
    )
    print(
        "  CT PhysicalVoxelCenterBoundsMm: "
        + _format_bounds(ct.physical_voxel_center_bounds_mm)
    )
    print(f"  PhysicalAABBOverlap: {'YES' if overlap else 'NO'}")
    if overlap_bounds is not None:
        print("  OverlapBoundsMm: " + _format_bounds(overlap_bounds))
    print(
        f"  PETCenterCoverageByCT: {covered_centers} / {total_centers} "
        f"({covered_centers / total_centers * 100.0:.6f}%)"
    )

    print("\n9 DETERMINISTIC SAMPLE POINTS")
    for index, point in enumerate(sample_points, start=1):
        print(
            f"  {index:02d}. {point['kind']} "
            f"PETVoxel={point['pet_voxel']} "
            f"WorldMm={_format_vector(point['world_coordinate_mm'])} "
            f"SourceCTVoxel={_format_vector(point['source_ct_voxel'])} "
            f"InsideCTFOV={'YES' if point['inside_ct_fov'] else 'NO'}"
        )

    print("\nCOARSE ALIGNMENT DIAGNOSTIC")
    print("  Descriptive only; thresholds do not alter images or determine pass/fail.")
    for threshold in PET_THRESHOLDS:
        label = f"suvbw_gt_{threshold:g}"
        extent = alignment["pet_extents"][label]
        line = f"  PET SUVbw > {threshold:g}: voxels={extent['voxel_count']}"
        if "physical_voxel_center_bounds_mm" in extent:
            bounds = extent["physical_voxel_center_bounds_mm"]
            line += "; PhysicalVoxelCenterBoundsMm: " + _format_bounds(
                (tuple(bounds["minimum"]), tuple(bounds["maximum"]))
            )
        print(line)
        comparison = alignment["pet_extent_vs_ct_tissue_extent"].get(label)
        if comparison is not None:
            print(
                "      vs CT tissue extent: "
                f"AABBOverlap={'YES' if comparison['ct_extent_aabb_overlap'] else 'NO'}; "
                "BoundingBoxCenterDistanceMm="
                f"{comparison['bounding_box_center_distance_mm']:.6f}"
            )
    ct_extent = alignment["ct_tissue_extent_hu_gt_minus_500"]
    line = f"  CT HU > -500: voxels={ct_extent['voxel_count']}"
    if "physical_voxel_center_bounds_mm" in ct_extent:
        bounds = ct_extent["physical_voxel_center_bounds_mm"]
        line += "; PhysicalVoxelCenterBoundsMm: " + _format_bounds(
            (tuple(bounds["minimum"]), tuple(bounds["maximum"]))
        )
    print(line)

    print("\nPOST-WRITE VERIFICATION")
    for name, passed in post_write_checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print("\nOUTPUTS")
    for path in OUTPUT_PATHS:
        print(f"  {path.relative_to(PROJECT_ROOT)}: {path.stat().st_size} bytes")

    print("\nWARNINGS")
    if outside_count:
        print(
            "  PET grid extends beyond the source CT voxel-center field of view; "
            f"{outside_count} CT output voxels ({outside_count / total_centers * 100.0:.6f}%) "
            f"were filled with {CT_OUTSIDE_FOV_HU:g} HU."
        )
    else:
        print("  None.")
    print("  Coarse threshold extents are descriptive and are not an alignment verdict.")


def prepare(*, overwrite: bool = False) -> None:
    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in existing)
        raise PreparationError(
            f"output already exists ({names}); rerun with --overwrite to replace it"
        )

    pet_image, pet = _load_image(SOURCE_PET, "PET")
    ct_image, ct = _load_image(SOURCE_CT, "CT")
    pet_data = np.asanyarray(pet_image.dataobj, dtype=np.float32)
    ct_data = np.asanyarray(ct_image.dataobj, dtype=np.float32)
    pet_statistics = _array_statistics(pet_data)
    source_ct_statistics = _array_statistics(ct_data)
    if pet_statistics.non_finite_voxels:
        raise PreparationError("PET source contains non-finite values")
    if source_ct_statistics.non_finite_voxels:
        raise PreparationError("CT source contains non-finite values")

    target_to_source = np.linalg.inv(ct.affine) @ pet.affine
    overlap, overlap_bounds = _aabb_overlap(
        pet.physical_voxel_center_bounds_mm,
        ct.physical_voxel_center_bounds_mm,
    )
    if not overlap:
        raise PreparationError("PET and CT physical voxel-center bounds do not overlap")
    sample_points = _sample_points(pet, ct, target_to_source)
    covered_centers, total_centers = _coverage_by_pet_centers(
        pet, ct, target_to_source
    )

    ct_on_pet_grid = affine_transform(
        ct_data,
        matrix=target_to_source[:3, :3],
        offset=target_to_source[:3, 3],
        output_shape=pet.shape,
        output=np.float32,
        order=1,
        mode="constant",
        cval=CT_OUTSIDE_FOV_HU,
        prefilter=False,
    )
    ct_statistics = _array_statistics(ct_on_pet_grid)
    if ct_statistics.non_finite_voxels:
        raise PreparationError("resampled CT contains non-finite values")
    alignment = _alignment_diagnostic(pet_data, ct_on_pet_grid, pet.affine)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary_pet = OUTPUT_DIRECTORY / ".study_0001.copying.nii.gz"
    temporary_ct = OUTPUT_DIRECTORY / ".study_0000.writing.nii.gz"
    temporary_paths = (temporary_pet, temporary_ct)
    try:
        shutil.copyfile(SOURCE_PET, temporary_pet)
        _write_ct(ct_on_pet_grid, pet_image, temporary_ct)

        source_pet_hash = _sha256(SOURCE_PET)
        source_ct_hash = _sha256(SOURCE_CT)
        output_pet_hash = _sha256(temporary_pet)
        output_ct_hash = _sha256(temporary_ct)
        post_write_checks, written_ct_statistics = _verify_outputs(
            SOURCE_PET,
            temporary_pet,
            temporary_ct,
            pet,
            ct_statistics,
            total_centers - covered_centers,
            target_to_source,
            ct,
        )
        os.replace(temporary_pet, OUTPUT_PET)
        os.replace(temporary_ct, OUTPUT_CT)
        if _sha256(OUTPUT_PET) != output_pet_hash or _sha256(OUTPUT_CT) != output_ct_hash:
            raise PreparationError("an output hash changed during atomic replacement")
        provenance = _provenance(
            pet=pet,
            ct=ct,
            source_pet_hash=source_pet_hash,
            source_ct_hash=source_ct_hash,
            output_pet_hash=output_pet_hash,
            output_ct_hash=output_ct_hash,
            pet_statistics=pet_statistics,
            ct_statistics=written_ct_statistics,
            target_to_source=target_to_source,
            overlap=overlap,
            overlap_bounds=overlap_bounds,
            covered_centers=covered_centers,
            total_centers=total_centers,
            sample_points=sample_points,
            alignment=alignment,
            post_write_checks=post_write_checks,
        )
        _write_json_atomic(OUTPUT_PROVENANCE, provenance)
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()

    _print_report(
        pet=pet,
        ct=ct,
        pet_statistics=pet_statistics,
        ct_statistics=written_ct_statistics,
        source_pet_hash=source_pet_hash,
        output_pet_hash=output_pet_hash,
        output_ct_hash=output_ct_hash,
        covered_centers=covered_centers,
        total_centers=total_centers,
        sample_points=sample_points,
        alignment=alignment,
        overlap=overlap,
        overlap_bounds=overlap_bounds,
        post_write_checks=post_write_checks,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare GLOW-FDG inputs by copying PET SUVbw unchanged and linearly "
            "resampling CT HU onto the PET grid. No registration is performed."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the three fixed outputs if they already exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        prepare(overwrite=arguments.overwrite)
    except PreparationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
