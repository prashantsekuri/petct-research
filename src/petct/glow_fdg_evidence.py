"""Write deterministic, non-clinical GLOW-FDG lesion evidence records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .localize_glow_fdg import (
    MAX_DISTANCE_MM,
    MAX_NEAREST,
    ANATOMY_DIR,
    LESION_PATH,
    PET_PATH,
    _aabb,
    _aabb_distance,
    _distance_field,
    _load,
    _mask_aabb,
    _orthogonal,
    _world_points,
)

CT_PATH = Path(__file__).resolve().parents[2] / "output/segmentation/glow_fdg/input/study_0000.nii.gz"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "output/evidence/glow_fdg_lesions.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_evidence() -> dict[str, object]:
    pet = _load(PET_PATH, "PET")
    ct = _load(CT_PATH, "prepared CT")
    lesion = _load(LESION_PATH, "ensemble lesion mask")
    if len(pet.shape) != 3 or lesion.shape != pet.shape:
        raise RuntimeError("PET and lesion mask must be matching 3-D grids")
    if not np.allclose(pet.affine, lesion.affine, rtol=0.0, atol=1e-5):
        raise RuntimeError("PET and lesion mask affines do not match")
    pet_data = np.asanyarray(pet.dataobj, dtype=np.float32)
    labels, count = ndimage.label(np.asanyarray(lesion.dataobj) != 0, np.ones((3, 3, 3), dtype=np.uint8))
    pet_affine = np.asarray(pet.affine, dtype=float)
    spacing = np.asarray(nib.affines.voxel_sizes(pet_affine), dtype=float)

    anatomy_records = []
    for path in sorted(ANATOMY_DIR.glob("*.nii.gz")):
        name = path.name[:-7]
        image = _load(path, f"anatomy mask {name}")
        mask = np.asanyarray(image.dataobj) != 0
        if mask.ndim != 3:
            continue
        affine = np.asarray(image.affine, dtype=float)
        box = _mask_aabb(mask, affine)
        if box is not None:
            anatomy_records.append((name, mask, affine, box, np.asarray(nib.affines.voxel_sizes(affine), dtype=float), _orthogonal(affine)))

    components = []
    for cid, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        local = np.argwhere(labels[slc] == cid)
        coords = local + np.asarray([s.start for s in slc])
        world = _world_points(pet_affine, coords)
        values = pet_data[tuple(coords.T)]
        centroid = world.mean(axis=0)
        components.append({"coords": coords, "world": world, "values": values, "box": _aabb(world), "centroid": centroid})
    components.sort(key=lambda c: (-len(c["coords"]), tuple(float(v) for v in c["centroid"])))

    lesions = []
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    for index, component in enumerate(components, start=1):
        coords, world, values, box, centroid = (component[k] for k in ("coords", "world", "values", "box", "centroid"))
        overlaps = []
        nearby = []
        vertebrae = []
        for name, mask, affine, anatomy_box, anatomy_spacing, is_orthogonal in anatomy_records:
            anatomy_idx = _world_points(np.linalg.inv(affine), world)
            sampled = ndimage.map_coordinates(mask.astype(np.uint8), anatomy_idx.T, order=0, mode="constant", cval=0)
            overlap_count = int(np.count_nonzero(sampled))
            if overlap_count:
                overlaps.append({"structure": name, "voxel_count": overlap_count, "percent": overlap_count / len(coords) * 100.0})
            if _aabb_distance(box, anatomy_box) <= MAX_DISTANCE_MM and is_orthogonal:
                distance, centroid_distance = _distance_field(mask, affine, world, anatomy_spacing)
                if np.isfinite(distance):
                    nearby.append({"structure": name, "distance_mm": float(distance)})
                    if "vertebra" in name.lower():
                        vertebrae.append((name, float(distance), float(centroid_distance)))
        vertebra = min(vertebrae, key=lambda item: item[1]) if vertebrae else None
        lesions.append({
            "candidate_id": f"GLOW_FDG_{index:03d}",
            "mask": {
                "voxel_count": int(len(coords)),
                "volume_ml": float(len(coords) * voxel_volume_ml),
                "voxel_bbox": {"min": [int(v) for v in coords.min(0)], "max": [int(v) for v in coords.max(0)]},
                "centroid_ras_mm": [float(v) for v in centroid],
            },
            "pet": {"suvmax": float(np.max(values)), "suvmean": float(np.mean(values, dtype=np.float64))},
            "anatomy": {
                "direct_overlap": sorted(overlaps, key=lambda x: x["structure"]),
                "nearest_structures": sorted(nearby, key=lambda x: (x["distance_mm"], x["structure"]))[:MAX_NEAREST],
                "nearest_vertebra": ({
                    "structure": vertebra[0],
                    "lesion_to_mask_distance_mm": vertebra[1],
                    "centroid_to_mask_distance_mm": vertebra[2],
                    "within_50mm": vertebra[1] <= MAX_DISTANCE_MM,
                } if vertebra else {
                    "structure": None,
                    "lesion_to_mask_distance_mm": None,
                    "centroid_to_mask_distance_mm": None,
                    "within_50mm": False,
                }),
            },
        })

    return {
        "schema_version": 1,
        "model": {"name": "GLOW-FDG", "inference": "5-fold ensemble", "folds": [0, 1, 2, 3, 4]},
        "reference_grid": {"shape": [int(v) for v in pet.shape], "affine": pet_affine.tolist(), "orientation": "RAS", "voxel_spacing_mm": [float(v) for v in spacing]},
        "provenance": {"pet_sha256": _sha256(PET_PATH), "ct_sha256": _sha256(CT_PATH), "segmentation_sha256": _sha256(LESION_PATH)},
        "analysis": {"connectivity": 26, "component_order": "voxel_count_descending_with_deterministic_tiebreak", "direct_overlap_method": "PET voxel centers sampled on anatomy grid with nearest-neighbor", "nearest_anatomy_max_distance_mm": MAX_DISTANCE_MM, "nearest_anatomy_count": MAX_NEAREST, "distance_method": "local spacing-aware EDT after physical AABB pruning"},
        "lesions": lesions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write GLOW-FDG lesion evidence JSON.")
    parser.parse_args()
    evidence = build_evidence()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
