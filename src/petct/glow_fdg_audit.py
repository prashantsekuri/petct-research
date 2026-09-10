"""Read-only connected-component audit for a GLOW-FDG lesion mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_PATH = PROJECT_ROOT / "output/segmentation/glow_fdg/prediction_fold0/study.nii.gz"
PET_PATH = PROJECT_ROOT / "output/segmentation/glow_fdg/input/study_0001.nii.gz"
MAX_COMPONENTS_TO_PRINT = 20


class AuditError(RuntimeError):
    """Raised when the fixed audit inputs cannot be measured safely."""


def _load(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    if not path.is_file():
        raise AuditError(f"{label} input is missing")
    try:
        return nib.load(path)
    except Exception as error:
        raise AuditError(f"unable to load {label} NIfTI: {error}") from error


def _physical_centroid(affine: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    voxel_centroid = coordinates.mean(axis=0)
    return (affine @ np.append(voxel_centroid, 1.0))[:3]


def _component_measurements(
    labels: np.ndarray,
    component_count: int,
    pet_data: np.ndarray,
    affine: np.ndarray,
    voxel_spacing: np.ndarray,
) -> list[dict[str, object]]:
    slices = ndimage.find_objects(labels)
    voxel_volume_ml = float(np.prod(voxel_spacing) / 1000.0)
    components: list[dict[str, object]] = []
    for component_id in range(1, component_count + 1):
        component_slice = slices[component_id - 1]
        if component_slice is None:
            continue
        local_labels = labels[component_slice]
        local_coordinates = np.argwhere(local_labels == component_id)
        if local_coordinates.size == 0:
            continue
        starts = np.asarray([axis.start for axis in component_slice], dtype=np.int64)
        coordinates = local_coordinates + starts
        values = pet_data[tuple(coordinates.T)]
        if not np.all(np.isfinite(values)):
            raise AuditError("PET contains non-finite values within a lesion component")
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        components.append(
            {
                "voxel_count": int(coordinates.shape[0]),
                "volume_ml": float(coordinates.shape[0] * voxel_volume_ml),
                "voxel_minimum": tuple(int(value) for value in minimum),
                "voxel_maximum": tuple(int(value) for value in maximum),
                "centroid_ras_mm": tuple(
                    float(value) for value in _physical_centroid(affine, coordinates)
                ),
                "suv_max": float(np.max(values)),
                "suv_mean": float(np.mean(values, dtype=np.float64)),
            }
        )
    components.sort(key=lambda item: int(item["voxel_count"]), reverse=True)
    return components


def audit() -> None:
    mask_image = _load(MASK_PATH, "lesion mask")
    pet_image = _load(PET_PATH, "reference PET")
    if len(mask_image.shape) != 3 or len(pet_image.shape) != 3:
        raise AuditError("mask and PET must both be three-dimensional")
    if mask_image.shape != pet_image.shape:
        raise AuditError(
            f"mask/PET shape mismatch: {mask_image.shape} versus {pet_image.shape}"
        )
    if not np.allclose(mask_image.affine, pet_image.affine, rtol=0.0, atol=1e-5):
        raise AuditError("mask/PET affine mismatch")

    mask_data = np.asanyarray(mask_image.dataobj)
    pet_data = np.asanyarray(pet_image.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(pet_data)):
        raise AuditError("reference PET contains non-finite values")
    lesion_mask = mask_data != 0
    total_lesion_voxels = int(np.count_nonzero(lesion_mask))
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, component_count = ndimage.label(lesion_mask, structure=structure)
    voxel_spacing = np.asarray(nib.affines.voxel_sizes(pet_image.affine), dtype=np.float64)
    components = _component_measurements(
        labels,
        int(component_count),
        pet_data,
        np.asarray(pet_image.affine, dtype=np.float64),
        voxel_spacing,
    )

    print(f"Total lesion voxels: {total_lesion_voxels}")
    print(f"Number of connected components: {len(components)}")
    printable = components[:MAX_COMPONENTS_TO_PRINT]
    for index, component in enumerate(printable, start=1):
        print(f"\nComponent {index}")
        print(f"  Voxel count: {component['voxel_count']}")
        print(f"  Approximate volume mL: {component['volume_ml']:.6f}")
        print(
            "  Voxel bounding box: "
            f"minimum={component['voxel_minimum']}, maximum={component['voxel_maximum']}"
        )
        print(
            "  Physical centroid RAS mm: "
            + "(" + ", ".join(f"{value:.6f}" for value in component["centroid_ras_mm"]) + ")"
        )
        print(f"  SUVmax: {component['suv_max']:.9g}")
        print(f"  SUVmean: {component['suv_mean']:.9g}")
    remaining = len(components) - len(printable)
    if remaining:
        remaining_voxels = sum(
            int(component["voxel_count"]) for component in components[MAX_COMPONENTS_TO_PRINT:]
        )
        print(
            f"\nRemaining components not printed: {remaining} "
            f"({remaining_voxels} lesion voxels)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the fixed GLOW-FDG fold-0 lesion mask without writing files."
    )
    parser.parse_args(argv)
    try:
        audit()
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
