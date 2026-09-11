"""Extract deterministic native-resolution CT ROIs for selected candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "output/evidence/glow_fdg_lesions_ct.json"
PET = ROOT / "output/nifti/PET_SUVbw.nii.gz"
CT = ROOT / "output/nifti/CT_WB_CECT.nii.gz"
CT_PROVENANCE = ROOT / "output/nifti/ct_provenance.json"
OUT = ROOT / "output/evidence/native_ct_rois"
TARGETS = ("GLOW_FDG_002", "GLOW_FDG_003", "GLOW_FDG_004", "GLOW_FDG_005", "GLOW_FDG_006")
TARGET_MM = 80.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".native_ct_roi.", dir=path.parent)
    try:
        mode = "wb" if isinstance(payload, bytes) else "w"
        with os.fdopen(fd, mode, encoding=None if mode == "wb" else "utf-8") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def run() -> dict[str, object]:
    evidence = json.loads(EVIDENCE.read_text())
    records = {x["candidate_id"]: x for x in evidence["lesions"]}
    if any(cid not in records for cid in TARGETS):
        raise RuntimeError("one or more requested candidates are absent from evidence")
    pet_img, ct_img = nib.load(PET), nib.load(CT)
    if not np.isfinite(np.linalg.det(ct_img.affine[:3, :3])):
        raise RuntimeError("native CT affine is not invertible")
    ct_inv = np.linalg.inv(ct_img.affine)
    pet_inv = np.linalg.inv(pet_img.affine)
    spacing = np.asarray(nib.affines.voxel_sizes(ct_img.affine), dtype=float)
    counts = np.ceil(TARGET_MM / spacing).astype(int)
    counts += counts % 2 == 0
    ct_data = np.asanyarray(ct_img.dataobj)
    source_hash = _sha(CT)
    provenance_expected = json.loads(CT_PROVENANCE.read_text())["output"]["sha256"]
    if source_hash != provenance_expected:
        raise RuntimeError("native CT SHA-256 does not match CT provenance")
    items = []
    for cid in TARGETS:
        world = np.asarray(records[cid]["mask"]["centroid_ras_mm"], dtype=float)
        pet_voxel = nib.affines.apply_affine(pet_inv, world)
        reconstructed_world = nib.affines.apply_affine(pet_img.affine, pet_voxel)
        if not np.allclose(reconstructed_world, world, atol=1e-4, rtol=0):
            raise RuntimeError(f"PET/world centroid mismatch for {cid}")
        native_float = nib.affines.apply_affine(ct_inv, world)
        if np.any(native_float < 0) or np.any(native_float > np.asarray(ct_img.shape) - 1):
            raise RuntimeError(f"centroid outside native CT voxel-center coverage for {cid}")
        center = np.rint(native_float).astype(int)
        starts = center - counts // 2
        ends = starts + counts
        clipped = False
        for axis, size in enumerate(ct_img.shape):
            if starts[axis] < 0:
                ends[axis] -= starts[axis]; starts[axis] = 0; clipped = True
            if ends[axis] > size:
                starts[axis] -= ends[axis] - size; ends[axis] = size; clipped = True
            starts[axis] = max(0, starts[axis]); ends[axis] = min(size, ends[axis])
        slices = tuple(slice(int(a), int(b)) for a, b in zip(starts, ends))
        roi = np.asanyarray(ct_data[slices]).copy()
        roi_affine = np.asarray(ct_img.affine, dtype=float).copy()
        roi_affine[:3, 3] = nib.affines.apply_affine(ct_img.affine, starts)
        corner = np.asarray([[i, j, k] for i in (0, roi.shape[0] - 1) for j in (0, roi.shape[1] - 1) for k in (0, roi.shape[2] - 1)], dtype=float)
        bounds = nib.affines.apply_affine(roi_affine, corner)
        folder = OUT / cid
        folder.mkdir(parents=True, exist_ok=True)
        nii_path = folder / "ct_roi.nii.gz"
        nib.save(nib.Nifti1Image(roi, roi_affine, header=ct_img.header.copy()), nii_path)
        reloaded = np.asanyarray(nib.load(nii_path).dataobj)
        if not np.array_equal(reloaded, roi):
            raise RuntimeError(f"ROI values are not an exact native CT subarray for {cid}")
        metadata = {"candidate_id": cid, "source_evidence_sha256": _sha(EVIDENCE), "native_ct_sha256": source_hash, "centroid_ras_mm": world.tolist(), "centroid_native_ct_voxel_float": native_float.tolist(), "centroid_native_ct_voxel_rounded": center.tolist(), "roi_voxel_bounds": {"min_inclusive": starts.tolist(), "max_exclusive": ends.tolist()}, "roi_physical_bounds_ras_mm_voxel_centers": {"min": bounds.min(0).tolist(), "max": bounds.max(0).tolist()}, "native_voxel_spacing_mm": spacing.tolist(), "roi_shape": [int(v) for v in roi.shape], "crop_clipped_at_image_boundary": clipped, "target_physical_extent_mm": [TARGET_MM] * 3, "voxel_count_convention": "odd counts via ceil(80mm/spacing); rounded centroid is central voxel when unclipped", "interpolation_or_resampling": False, "exact_native_subarray_verified": True}
        _atomic(folder / "roi_metadata.json", json.dumps(metadata, indent=2) + "\n")
        items.append({"candidate_id": cid, "roi_shape": metadata["roi_shape"], "centroid_ras_mm": world.tolist(), "ct_coverage": "inside_native_ct_voxel_center_coverage", "ct_roi_sha256": _sha(nii_path), "metadata_sha256": _sha(folder / "roi_metadata.json")})
    index = {"source_evidence_sha256": _sha(EVIDENCE), "native_ct_sha256": source_hash, "target_physical_extent_mm": [TARGET_MM] * 3, "voxel_count_convention": "odd counts via ceil(80mm/spacing)", "rois": items}
    _atomic(OUT / "index.json", json.dumps(index, indent=2) + "\n")
    return index


if __name__ == "__main__":
    argparse.ArgumentParser(description="Extract selected native CT ROIs.").parse_args()
    result = run()
    print(json.dumps(result, indent=2))
