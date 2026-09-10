"""Print-only anatomical localization of GLOW-FDG lesion components."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PET_PATH = PROJECT_ROOT / "output/segmentation/glow_fdg/input/study_0001.nii.gz"
LESION_PATH = PROJECT_ROOT / "output/segmentation/glow_fdg/prediction/study.nii.gz"
ANATOMY_DIR = PROJECT_ROOT / "output/segmentation/totalsegmentator_fast"
MAX_DISTANCE_MM = 50.0
MAX_NEAREST = 5


class LocalizationError(RuntimeError):
    pass


def _load(path: Path, label: str):
    if not path.is_file():
        raise LocalizationError(f"{label} input is missing")
    try:
        return nib.load(path)
    except Exception as error:
        raise LocalizationError(f"unable to load {label}: {error}") from error


def _world_points(affine: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(affine, indices)


def _aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(points, axis=0), np.max(points, axis=0)


def _aabb_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    gap = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(gap))


def _mask_aabb(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if not len(coords):
        return None
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    corners = np.asarray(
        [[i, j, k] for i in (lo[0], hi[0]) for j in (lo[1], hi[1]) for k in (lo[2], hi[2])],
        dtype=float,
    )
    return _aabb(_world_points(affine, corners))


def _orthogonal(affine: np.ndarray) -> bool:
    axes = affine[:3, :3]
    norms = np.linalg.norm(axes, axis=0)
    if np.any(norms == 0):
        return False
    unit = axes / norms
    return bool(np.allclose(unit.T @ unit, np.eye(3), rtol=0.0, atol=1e-4))


def _distance_field(
    mask: np.ndarray,
    affine: np.ndarray,
    lesion_world: np.ndarray,
    spacing: np.ndarray,
) -> tuple[float, float]:
    """Return minimum lesion distance and centroid distance using a local ROI."""
    inv = np.linalg.inv(affine)
    lesion_idx = _world_points(inv, lesion_world)
    margin = np.ceil(MAX_DISTANCE_MM / spacing).astype(int) + 2
    lo = np.floor(np.min(lesion_idx, axis=0)).astype(int) - margin
    hi = np.ceil(np.max(lesion_idx, axis=0)).astype(int) + margin + 1
    shape = np.asarray(mask.shape)
    crop_lo = np.maximum(lo, 0)
    crop_hi = np.minimum(hi, shape)
    if np.any(crop_lo >= crop_hi):
        return float("inf"), float("inf")
    local = mask[tuple(slice(int(a), int(b)) for a, b in zip(crop_lo, crop_hi))]
    if not np.any(local):
        return float("inf"), float("inf")
    distance = ndimage.distance_transform_edt(~local, sampling=spacing)
    query = lesion_idx - crop_lo
    sampled = ndimage.map_coordinates(distance, query.T, order=1, mode="constant", cval=np.inf)
    centroid_idx = _world_points(inv, lesion_world.mean(axis=0)[None, :])[0] - crop_lo
    centroid_distance = float(
        ndimage.map_coordinates(distance, centroid_idx[:, None], order=1, mode="constant", cval=np.inf)[0]
    )
    return float(np.min(sampled)), centroid_distance


def localize() -> None:
    pet = _load(PET_PATH, "PET")
    lesion = _load(LESION_PATH, "lesion mask")
    if len(pet.shape) != 3 or len(lesion.shape) != 3 or pet.shape != lesion.shape:
        raise LocalizationError("PET and lesion mask must be matching 3-D grids")
    if not np.allclose(pet.affine, lesion.affine, rtol=0.0, atol=1e-5):
        raise LocalizationError("PET and lesion mask affines do not match")
    pet_data = np.asanyarray(pet.dataobj, dtype=np.float32)
    labels, count = ndimage.label(np.asanyarray(lesion.dataobj) != 0, np.ones((3, 3, 3), dtype=np.uint8))
    spacing = np.asarray(nib.affines.voxel_sizes(pet.affine), dtype=float)
    anatomy_paths = sorted(ANATOMY_DIR.glob("*.nii.gz"))
    if not anatomy_paths:
        raise LocalizationError("no TotalSegmentator masks found")
    anatomy_records = []
    for path in anatomy_paths:
        name = path.name[:-7]
        image = _load(path, f"anatomy mask {name}")
        mask = np.asanyarray(image.dataobj) != 0
        if mask.ndim != 3:
            continue
        anatomy_affine = np.asarray(image.affine, dtype=float)
        anatomy_box = _mask_aabb(mask, anatomy_affine)
        if anatomy_box is None:
            continue
        anatomy_records.append(
            (
                name,
                mask,
                anatomy_affine,
                anatomy_box,
                np.asarray(nib.affines.voxel_sizes(anatomy_affine), dtype=float),
                _orthogonal(anatomy_affine),
            )
        )
    nonorthogonal = sum(not record[5] for record in anatomy_records)
    if nonorthogonal:
        print(
            f"Distance limitation: {nonorthogonal} anatomy masks have non-orthogonal affines; "
            "spacing-only local EDT distances are omitted for those masks.\n"
        )
    print("DirectOverlap uses transformed PET voxel centers sampled with nearest-neighbor anatomy voxels; it is approximate.\n")

    components = []
    for cid, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        local = np.argwhere(labels[slc] == cid)
        coords = local + np.asarray([s.start for s in slc])
        world = _world_points(pet.affine, coords)
        values = pet_data[tuple(coords.T)]
        components.append((cid, coords, world, values, _aabb(world)))

    for rank, (cid, coords, world, values, lesion_box) in enumerate(
        sorted(components, key=lambda item: len(item[1]), reverse=True), start=1
    ):
        print(f"COMPONENT {rank}")
        print(f"  Voxels: {len(coords)}")
        print(f"  VolumeMl: {len(coords) * float(np.prod(spacing)) / 1000.0:.6f}")
        centroid = world.mean(axis=0)
        print("  CentroidRAS: (" + ", ".join(f"{v:.3f}" for v in centroid) + ")")
        print(f"  SUVmax: {np.max(values):.7g}")
        print(f"  SUVmean: {np.mean(values, dtype=np.float64):.7g}")
        print(
            "  VoxelBoundingBox: "
            f"minimum={tuple(int(v) for v in coords.min(0))}, "
            f"maximum={tuple(int(v) for v in coords.max(0))}"
        )
        overlaps, nearby, vertebrae = [], [], []
        for name, mask, anatomy_affine, anatomy_box, anatomy_spacing, is_orthogonal in anatomy_records:
            inv = np.linalg.inv(anatomy_affine)
            anatomy_idx = _world_points(inv, world)
            sampled = ndimage.map_coordinates(mask.astype(np.uint8), anatomy_idx.T, order=0, mode="constant", cval=0)
            overlap_count = int(np.count_nonzero(sampled))
            if overlap_count:
                overlaps.append((name, overlap_count, overlap_count / len(coords) * 100.0))
            lower_bound = _aabb_distance(lesion_box, anatomy_box)
            if lower_bound <= MAX_DISTANCE_MM and is_orthogonal:
                distance, centroid_distance = _distance_field(mask, anatomy_affine, world, anatomy_spacing)
                if np.isfinite(distance):
                    nearby.append((name, distance))
                    if "vertebra" in name.lower():
                        vertebrae.append((name, distance, centroid_distance))
        print("  DirectOverlap:")
        if overlaps:
            for name, voxels, percent in sorted(overlaps):
                print(f"    {name}: {voxels} ({percent:.2f}%)")
        else:
            print("    none (voxel-center approximation)")
        print("  NearestAnatomy:")
        for name, distance in sorted(nearby, key=lambda item: item[1])[:MAX_NEAREST]:
            print(f"    {name}: {distance:.2f} mm")
        print("  NearestVertebra:")
        if vertebrae:
            name, distance, centroid_distance = min(vertebrae, key=lambda item: item[1])
            print(f"    {name}")
            print(f"    LesionToMaskDistanceMm: {distance:.2f}")
            print(f"    CentroidToMaskDistanceMm: {centroid_distance:.2f}")
        else:
            print("    none within 50 mm")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Localize GLOW-FDG lesions against TotalSegmentator masks.")
    parser.parse_args(argv)
    try:
        localize()
    except LocalizationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
