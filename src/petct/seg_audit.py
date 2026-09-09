"""Audit TotalSegmentator masks against the derived CT NIfTI geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


AFFINE_ATOL = 1e-5
CT_RELATIVE_PATH = Path("output/nifti/CT_WB_CECT.nii.gz")
MASKS_RELATIVE_PATH = Path("output/segmentation/totalsegmentator_fast")


class AuditError(RuntimeError):
    """Raised when the segmentation audit cannot be completed."""


@dataclass(frozen=True)
class CTReference:
    shape: tuple[int, ...]
    affine: np.ndarray
    voxel_spacing_mm: tuple[float, ...]
    orientation: tuple[str, ...]


@dataclass(frozen=True)
class MaskResult:
    name: str
    status: str
    shape: tuple[int, ...]
    nonzero_voxels: int
    voxel_bounds: tuple[tuple[int, ...], tuple[int, ...]] | None
    physical_voxel_center_bounds_mm: (
        tuple[tuple[float, ...], tuple[float, ...]] | None
    )
    maximum_affine_difference: float


@dataclass(frozen=True)
class AuditReport:
    ct: CTReference
    masks: tuple[MaskResult, ...]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_nifti(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    if not path.is_file():
        raise AuditError(f"{label} does not exist: {path}")
    try:
        return nib.load(path)
    except Exception as error:
        raise AuditError(f"failed to load {label}: {error}") from error


def _orientation(affine: np.ndarray) -> tuple[str, ...]:
    return tuple(str(code) for code in nib.aff2axcodes(affine))


def _voxel_bounds(data: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    occupied_axes = []
    for axis in range(data.ndim):
        reduction_axes = tuple(index for index in range(data.ndim) if index != axis)
        occupied = np.flatnonzero(np.any(data != 0, axis=reduction_axes))
        occupied_axes.append((int(occupied[0]), int(occupied[-1])))
    return (
        tuple(bounds[0] for bounds in occupied_axes),
        tuple(bounds[1] for bounds in occupied_axes),
    )


def _physical_voxel_center_bounds(
    voxel_bounds: tuple[tuple[int, ...], tuple[int, ...]], affine: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(voxel_bounds[0]) != 3:
        raise AuditError("physical bounding boxes require three-dimensional masks")

    minimum, maximum = voxel_bounds
    corners = np.asarray(
        [
            (axis_0, axis_1, axis_2)
            for axis_0 in (minimum[0], maximum[0])
            for axis_1 in (minimum[1], maximum[1])
            for axis_2 in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )
    physical_corners = nib.affines.apply_affine(affine, corners)
    return (
        tuple(float(value) for value in physical_corners.min(axis=0)),
        tuple(float(value) for value in physical_corners.max(axis=0)),
    )


def _audit_mask(path: Path, ct: CTReference) -> MaskResult:
    image = _load_nifti(path, f"mask {path.name}")
    shape = tuple(int(value) for value in image.shape)
    affine = np.asarray(image.affine, dtype=np.float64)
    maximum_affine_difference = float(np.max(np.abs(affine - ct.affine)))
    affine_matches = np.allclose(
        affine,
        ct.affine,
        rtol=0.0,
        atol=AFFINE_ATOL,
    )

    try:
        data = np.asanyarray(image.dataobj)
    except Exception as error:
        raise AuditError(f"failed to read mask {path.name}: {error}") from error

    nonzero_voxels = int(np.count_nonzero(data))
    voxel_bounds = None
    physical_bounds = None
    if nonzero_voxels:
        voxel_bounds = _voxel_bounds(data)
        physical_bounds = _physical_voxel_center_bounds(voxel_bounds, affine)

    if shape != ct.shape:
        status = "SHAPE_MISMATCH"
    elif not affine_matches:
        status = "AFFINE_MISMATCH"
    elif nonzero_voxels == 0:
        status = "EMPTY"
    else:
        status = "PASS"

    return MaskResult(
        name=path.name,
        status=status,
        shape=shape,
        nonzero_voxels=nonzero_voxels,
        voxel_bounds=voxel_bounds,
        physical_voxel_center_bounds_mm=physical_bounds,
        maximum_affine_difference=maximum_affine_difference,
    )


def audit_segmentations() -> AuditReport:
    root = _project_root()
    ct_path = root / CT_RELATIVE_PATH
    masks_path = root / MASKS_RELATIVE_PATH

    ct_image = _load_nifti(ct_path, "CT reference")
    ct_affine = np.asarray(ct_image.affine, dtype=np.float64)
    ct = CTReference(
        shape=tuple(int(value) for value in ct_image.shape),
        affine=ct_affine,
        voxel_spacing_mm=tuple(
            float(value) for value in nib.affines.voxel_sizes(ct_affine)
        ),
        orientation=_orientation(ct_affine),
    )

    if not masks_path.is_dir():
        raise AuditError(f"masks directory does not exist: {masks_path}")
    mask_paths = sorted(masks_path.glob("*.nii.gz"), key=lambda path: path.name)
    if not mask_paths:
        raise AuditError(f"no .nii.gz masks found in: {masks_path}")

    return AuditReport(
        ct=ct,
        masks=tuple(_audit_mask(path, ct) for path in mask_paths),
    )


def _format_tuple(values: Sequence[float | int]) -> str:
    return "(" + ", ".join(f"{value:.12g}" for value in values) + ")"


def _format_bounds(
    bounds: tuple[tuple[float | int, ...], tuple[float | int, ...]]
) -> str:
    return f"minimum={_format_tuple(bounds[0])}, maximum={_format_tuple(bounds[1])}"


def _format_mask(mask: MaskResult) -> str:
    details = [
        f"{mask.name}: {mask.status}",
        f"voxels={mask.nonzero_voxels}",
        f"shape={mask.shape}",
    ]
    if mask.status == "AFFINE_MISMATCH":
        details.append(f"maximum_affine_difference={mask.maximum_affine_difference:.12g}")
    if mask.nonzero_voxels:
        assert mask.voxel_bounds is not None
        assert mask.physical_voxel_center_bounds_mm is not None
        details.append(f"VoxelBounds={_format_bounds(mask.voxel_bounds)}")
        details.append(
            "PhysicalVoxelCenterBoundsMm="
            f"{_format_bounds(mask.physical_voxel_center_bounds_mm)}"
        )
    return "  " + "; ".join(details)


def format_report(report: AuditReport) -> str:
    counts = {
        status: sum(mask.status == status for mask in report.masks)
        for status in ("PASS", "EMPTY", "SHAPE_MISMATCH", "AFFINE_MISMATCH")
    }
    mismatches = counts["SHAPE_MISMATCH"] + counts["AFFINE_MISMATCH"]

    lines = [
        "CT REFERENCE",
        f"  Shape: {report.ct.shape}",
        f"  VoxelSpacingMm: {_format_tuple(report.ct.voxel_spacing_mm)}",
        f"  Orientation: {''.join(report.ct.orientation)}",
        "  Affine:",
    ]
    lines.extend(
        "    " + " ".join(f"{value:.12g}" for value in row)
        for row in report.ct.affine
    )
    lines.extend(
        [
            "",
            "MASK SUMMARY",
            f"  TotalMasks: {len(report.masks)}",
            f"  PASS: {counts['PASS']}",
            f"  EMPTY: {counts['EMPTY']}",
            f"  SHAPE_MISMATCH: {counts['SHAPE_MISMATCH']}",
            f"  AFFINE_MISMATCH: {counts['AFFINE_MISMATCH']}",
            f"  Mismatches: {mismatches}",
            "",
            "NON-PASS MASKS",
        ]
    )
    non_pass = [mask for mask in report.masks if mask.status != "PASS"]
    lines.extend(_format_mask(mask) for mask in non_pass)
    if not non_pass:
        lines.append("  None")

    lines.extend(["", "REPRESENTATIVE PASS MASKS"])
    passing = [mask for mask in report.masks if mask.status == "PASS"]
    lines.extend(_format_mask(mask) for mask in passing[:3])
    if not passing:
        lines.append("  None")

    lines.extend(["", "TOP 20 LARGEST MASKS"])
    largest = sorted(
        report.masks,
        key=lambda mask: (-mask.nonzero_voxels, mask.name),
    )[:20]
    lines.extend(
        f"  {rank}. {mask.name}: {mask.nonzero_voxels} voxels [{mask.status}]"
        for rank, mask in enumerate(largest, start=1)
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Audit TotalSegmentator masks against the fixed derived CT reference."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    try:
        report = audit_segmentations()
    except AuditError as error:
        parser.exit(1, f"segmentation audit failed: {error}\n")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
