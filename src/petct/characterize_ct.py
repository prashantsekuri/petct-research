"""Deterministic CT characterization for canonical GLOW-FDG evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, distance

from .localize_glow_fdg import _load, _world_points

ROOT = Path(__file__).resolve().parents[2]
PET_PATH = ROOT / "output/segmentation/glow_fdg/input/study_0001.nii.gz"
CT_PATH = ROOT / "output/segmentation/glow_fdg/input/study_0000.nii.gz"
SEG_PATH = ROOT / "output/segmentation/glow_fdg/prediction/study.nii.gz"
EVIDENCE_PATH = ROOT / "output/evidence/glow_fdg_lesions.json"
PREP_PATH = ROOT / "output/segmentation/glow_fdg/input/preprocessing_provenance.json"
OUTPUT_PATH = ROOT / "output/evidence/glow_fdg_lesions_ct.json"
RADIUS_MM = 10.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {k: None for k in ("min", "max", "mean", "median", "std", "p05", "p25", "p75", "p95")}
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: None for k in ("min", "max", "mean", "median", "std", "p05", "p25", "p75", "p95")}
    return {"min": float(np.min(values)), "max": float(np.max(values)), "mean": float(np.mean(values, dtype=np.float64)), "median": float(np.median(values)), "std": float(np.std(values, dtype=np.float64)), "p05": float(np.percentile(values, 5)), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "p95": float(np.percentile(values, 95))}


def _bands(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {k: None for k in ("very_low_density_fraction", "fat_range_fraction", "soft_tissue_range_fraction", "high_density_fraction", "very_high_density_fraction")}
    return {
        "very_low_density_fraction": float(np.mean(values < -500)),
        "fat_range_fraction": float(np.mean((values >= -500) & (values < -30))),
        "soft_tissue_range_fraction": float(np.mean((values >= -30) & (values < 150))),
        "high_density_fraction": float(np.mean((values >= 150) & (values < 700))),
        "very_high_density_fraction": float(np.mean(values >= 700)),
    }


def _component_diameter(world: np.ndarray) -> float | None:
    # Convex hull is built from boundary voxel centers, avoiding interior points.
    return_value = None
    if len(world) >= 4:
        # Boundary extraction is performed in voxel space by the caller; this is already a compact set.
        try:
            hull = ConvexHull(world)
            vertices = world[hull.vertices]
            if len(vertices) >= 2:
                return_value = float(np.max(distance.pdist(vertices)))
        except Exception:
            return_value = None
    return return_value


def _write_atomic(payload: dict[str, object]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".glow_fdg_lesions_ct.", suffix=".json", dir=OUTPUT_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, OUTPUT_PATH)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def build() -> dict[str, object]:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    prep = json.loads(PREP_PATH.read_text(encoding="utf-8"))
    pet_img, ct_img, seg_img = _load(PET_PATH, "PET"), _load(CT_PATH, "CT"), _load(SEG_PATH, "segmentation")
    pet = np.asanyarray(pet_img.dataobj, dtype=np.float32)
    ct = np.asanyarray(ct_img.dataobj, dtype=np.float32)
    seg = np.asanyarray(seg_img.dataobj) != 0
    labels, n = ndimage.label(seg, np.ones((3, 3, 3), dtype=np.uint8))
    if n != len(evidence["lesions"]):
        raise RuntimeError("segmentation component count differs from evidence")
    pet_affine = np.asarray(pet_img.affine, dtype=float)
    spacing = np.asarray(nib.affines.voxel_sizes(pet_affine), dtype=float)
    source_shape = np.asarray(prep["inputs"]["ct"]["shape"], dtype=int)
    source_affine = np.asarray(prep["inputs"]["ct"]["affine"], dtype=float)
    source_inv = np.linalg.inv(source_affine)
    all_lesion = seg

    component_items = []
    for idx, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        local_labels = labels[slc]
        coords_local = np.argwhere(local_labels == idx)
        origin = np.asarray([s.start for s in slc])
        coords = coords_local + origin
        world = _world_points(pet_affine, coords)
        component_items.append((idx, slc, coords_local, coords, world, world.mean(axis=0)))
    component_items.sort(key=lambda item: (-len(item[3]), tuple(float(v) for v in item[5])))

    for record, item in zip(evidence["lesions"], component_items):
        idx, slc, coords_local, coords, world, _ = item
        if slc is None:
            raise RuntimeError("empty component unexpectedly present")
        origin = np.asarray([s.start for s in slc])
        coords = coords_local + origin
        world = _world_points(pet_affine, coords)
        source_coords = _world_points(source_inv, world)
        covered = np.all((source_coords >= 0) & (source_coords <= (source_shape - 1)), axis=1)
        ct_values = ct[tuple(coords.T)]
        valid_values = ct_values[covered & np.isfinite(ct_values)]

        margin = np.ceil(RADIUS_MM / spacing).astype(int) + 1
        lo = np.maximum(coords.min(0) - margin, 0)
        hi = np.minimum(coords.max(0) + margin + 1, np.asarray(pet.shape))
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        local_labels = labels[slc]
        component_local = labels[slices] == idx
        neighborhood_distance = ndimage.distance_transform_edt(~component_local, sampling=spacing)
        any_lesion_local = all_lesion[slices]
        neighborhood = (neighborhood_distance > 0) & (neighborhood_distance <= RADIUS_MM) & ~any_lesion_local
        neighborhood_global = np.zeros(pet.shape, dtype=bool)
        neighborhood_global[slices] = neighborhood
        neighbor_ct = ct[neighborhood_global]
        neighbor_source = _world_points(source_inv, _world_points(pet_affine, np.argwhere(neighborhood_global)))
        neighbor_covered = np.all((neighbor_source >= 0) & (neighbor_source <= (source_shape - 1)), axis=1)
        neighbor_values = neighbor_ct[neighbor_covered & np.isfinite(neighbor_ct)]

        center = world.mean(axis=0)
        center_ct_idx = _world_points(np.linalg.inv(ct_img.affine), center[None, :])[0]
        center_hu = float(ndimage.map_coordinates(ct, center_ct_idx[:, None], order=1, mode="constant", cval=np.nan)[0])
        if not np.all(np.isfinite(center_ct_idx)) or np.any(center_ct_idx < 0) or np.any(center_ct_idx > np.asarray(ct.shape) - 1):
            center_hu = None

        center_min, center_max = world.min(0), world.max(0)
        center_span = center_max - center_min
        bbox_extent = center_span + spacing
        component_mask_local = local_labels == idx
        boundary_local = coords_local[component_mask_local[tuple(coords_local.T)] & ~ndimage.binary_erosion(component_mask_local)[tuple(coords_local.T)]]
        diameter = _component_diameter(_world_points(pet_affine, boundary_local + origin))
        record["ct"] = {
            "coverage": {"total_candidate_voxels": int(len(coords)), "valid_ct_voxels": int(len(valid_values)), "outside_source_ct_fov_voxels": int(len(coords) - np.count_nonzero(covered)), "valid_ct_fraction": float(len(valid_values) / len(coords))},
            "intensity_hu": _stats(valid_values),
            "hu_bands": _bands(valid_values),
            "centroid_ct_hu": center_hu,
            "dimensions_mm": {"voxel_center_span_x": float(center_span[0]), "voxel_center_span_y": float(center_span[1]), "voxel_center_span_z": float(center_span[2]), "bbox_extent_x": float(bbox_extent[0]), "bbox_extent_y": float(bbox_extent[1]), "bbox_extent_z": float(bbox_extent[2]), "max_axis_aligned_extent": float(np.max(bbox_extent)), "approx_max_3d_diameter_mm": diameter},
            "neighborhood_10mm": {"valid_voxel_count": int(len(neighbor_values)), "hu_mean": _stats(neighbor_values)["mean"], "hu_median": _stats(neighbor_values)["median"], "hu_std": _stats(neighbor_values)["std"], "hu_bands": _bands(neighbor_values)},
        }
    result = dict(evidence)
    result["ct_characterization"] = {"source_ct_sha256": _sha(CT_PATH), "source_evidence_sha256": _sha(EVIDENCE_PATH), "outside_fov_method": "PET-grid voxel centers transformed through original CT affine; covered when 0 <= coordinate <= shape-1 (source CT voxel-center coverage for preprocessing linear interpolation)", "neighborhood_radius_mm": RADIUS_MM, "approx_max_3d_diameter_method": "maximum Euclidean separation of convex-hull boundary voxel centers (center-to-center approximation, not a RECIST measurement)", "hu_bands": {"very_low_density": "HU < -500", "fat_range": "-500 <= HU < -30", "soft_tissue_range": "-30 <= HU < 150", "high_density": "150 <= HU < 700", "very_high_density": "HU >= 700"}}
    return result


def main() -> int:
    argparse.ArgumentParser(description="Write deterministic CT characterization evidence.").parse_args()
    _write_atomic(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
