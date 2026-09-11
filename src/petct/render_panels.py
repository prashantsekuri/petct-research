"""Deterministic PET/CT visual review panels for GLOW-FDG candidates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
PET_PATH = ROOT / "output/segmentation/glow_fdg/input/study_0001.nii.gz"
CT_PATH = ROOT / "output/segmentation/glow_fdg/input/study_0000.nii.gz"
SEG_PATH = ROOT / "output/segmentation/glow_fdg/prediction/study.nii.gz"
EVIDENCE_PATH = ROOT / "output/evidence/glow_fdg_lesions_ct.json"
OUT_DIR = ROOT / "output/evidence/panels"
HALF_CROP_MM = 50.0
CT_VMIN, CT_VMAX = -160.0, 240.0
PET_VMIN, PET_VMAX = 0.0, 10.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _crop(center: int, size: int, half: int) -> slice:
    width = min(size, max(1, 2 * half + 1))
    start = center - width // 2
    start = max(0, min(start, size - width))
    return slice(start, start + width)


def _plane(data: np.ndarray, mask: np.ndarray, orient: str, indices: tuple[int, int, int]):
    x, y, z = indices
    if orient == "axial":
        return data[:, :, z].T, mask[:, :, z].T, ("R", "L", "P", "A")
    if orient == "coronal":
        return data[:, y, :].T, mask[:, y, :].T, ("R", "L", "I", "S")
    return data[x, :, :].T, mask[x, :, :].T, ("P", "A", "I", "S")


def _metadata_text(record: dict[str, object]) -> str:
    ct = record["ct"]
    anatomy = record["anatomy"]
    overlap = anatomy["direct_overlap"]
    overlap_text = ", ".join(f"{x['structure']} ({x['percent']:.1f}%)" for x in overlap) or "none"
    nearest = ", ".join(f"{x['structure']} ({x['distance_mm']:.1f} mm)" for x in anatomy["nearest_structures"][:3]) or "none"
    vertebra = anatomy["nearest_vertebra"]
    vertebra_text = "none" if vertebra["structure"] is None else f"{vertebra['structure']} ({vertebra['lesion_to_mask_distance_mm']:.1f} mm)"
    dims = ct["dimensions_mm"]
    return (f"{record['candidate_id']}\nSUVmax {record['pet']['suvmax']:.4g} | SUVmean {record['pet']['suvmean']:.4g} | "
            f"Volume {record['mask']['volume_ml']:.3f} mL\n"
            f"Dimensions (bbox) {dims['bbox_extent_x']:.1f} x {dims['bbox_extent_y']:.1f} x {dims['bbox_extent_z']:.1f} mm\n"
            f"Centroid RAS ({', '.join(f'{v:.1f}' for v in record['mask']['centroid_ras_mm'])})\n"
            f"Direct overlap: {overlap_text}\nNearest: {nearest}\nNearest vertebra: {vertebra_text}")


def render() -> dict[str, object]:
    start = time.perf_counter()
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    pet_img, ct_img, seg_img = (nib.load(p) for p in (PET_PATH, CT_PATH, SEG_PATH))
    if pet_img.shape != ct_img.shape or pet_img.shape != seg_img.shape or not np.allclose(pet_img.affine, ct_img.affine, atol=1e-5) or not np.allclose(pet_img.affine, seg_img.affine, atol=1e-5):
        raise RuntimeError("PET, CT, and segmentation geometry mismatch")
    pet = np.asanyarray(pet_img.dataobj, dtype=np.float32)
    ct = np.asanyarray(ct_img.dataobj, dtype=np.float32)
    seg = np.asanyarray(seg_img.dataobj) != 0
    labels, count = ndimage.label(seg, np.ones((3, 3, 3), dtype=np.uint8))
    components = []
    for cid, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        local = np.argwhere(labels[slc] == cid)
        origin = np.asarray([s.start for s in slc])
        coords = local + origin
        world = nib.affines.apply_affine(pet_img.affine, coords)
        components.append((cid, slc, coords, world.mean(axis=0)))
    components.sort(key=lambda item: (-len(item[2]), tuple(float(v) for v in item[3])))
    if len(components) != len(evidence["lesions"]):
        raise RuntimeError("component count differs from evidence")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_hashes = []
    spacing = np.asarray(nib.affines.voxel_sizes(pet_img.affine), dtype=float)
    half_vox = np.ceil(HALF_CROP_MM / spacing).astype(int)
    for record, (cid, slc, coords, centroid) in zip(evidence["lesions"], components):
        if not np.allclose(centroid, record["mask"]["centroid_ras_mm"], atol=1e-4, rtol=0):
            raise RuntimeError(f"centroid mismatch for {record['candidate_id']}")
        center = np.rint(nib.affines.apply_affine(np.linalg.inv(pet_img.affine), centroid)).astype(int)
        xs, ys, zs = (_crop(int(center[i]), pet.shape[i], int(half_vox[i])) for i in range(3))
        crop = (xs, ys, zs)
        fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=False)
        fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.18, wspace=0.04, hspace=0.12)
        planes = (("axial", (center[0], center[1], center[2])), ("coronal", (center[0], center[1], center[2])), ("sagittal", (center[0], center[1], center[2])))
        for col, (orient, indices) in enumerate(planes):
            ct_plane, _, labels_text = _plane(ct[crop], labels[crop] == cid, orient, tuple(np.asarray(center) - np.asarray([xs.start, ys.start, zs.start])))
            pet_plane, _, _ = _plane(pet[crop], labels[crop] == cid, orient, tuple(np.asarray(center) - np.asarray([xs.start, ys.start, zs.start])))
            mask_plane, _, _ = _plane((labels[crop] == cid).astype(float), labels[crop] == cid, orient, tuple(np.asarray(center) - np.asarray([xs.start, ys.start, zs.start])))
            axes[0, col].imshow(ct_plane, cmap="gray", vmin=CT_VMIN, vmax=CT_VMAX, origin="lower")
            axes[1, col].imshow(np.clip(pet_plane, PET_VMIN, PET_VMAX), cmap="inferno", vmin=PET_VMIN, vmax=PET_VMAX, origin="lower")
            axes[2, col].imshow(ct_plane, cmap="gray", vmin=CT_VMIN, vmax=CT_VMAX, origin="lower")
            axes[2, col].imshow(np.clip(pet_plane, PET_VMIN, PET_VMAX), cmap="inferno", vmin=PET_VMIN, vmax=PET_VMAX, alpha=0.55, origin="lower")
            axes[3, col].imshow(ct_plane, cmap="gray", vmin=CT_VMIN, vmax=CT_VMAX, origin="lower")
            axes[3, col].contour(mask_plane, levels=[0.5], colors="#00ff66", linewidths=1.5)
            for row in range(4):
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(orient.upper(), fontsize=11)
            axes[3, col].text(0.02, 0.02, f"{labels_text[0]} / {labels_text[1]}   {labels_text[2]} / {labels_text[3]}", transform=axes[3, col].transAxes, color="white", fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
        for row, label in enumerate(("CT", "PET SUVbw", "FUSED", "CANDIDATE CONTOUR")):
            axes[row, 0].set_ylabel(label, fontsize=9)
        fig.suptitle(_metadata_text(record) + "\nPET display: 0–10 SUV (values above 10 clipped for display only)", fontsize=10, ha="left", x=0.04)
        fig.text(0.04, 0.04, "RAS anatomical convention; crop is 100 mm per displayed axis where image bounds permit.", fontsize=8)
        out = OUT_DIR / f"{record['candidate_id']}.png"
        fig.savefig(out, dpi=120, facecolor="white")
        plt.close(fig)
        panel_hashes.append({"candidate_id": record["candidate_id"], "filename": out.name, "sha256": _sha(out)})
    index = [{"candidate_id": x["candidate_id"], "png": x["filename"], "sha256": x["sha256"]} for x in panel_hashes]
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    provenance = {"inputs": {"pet_sha256": _sha(PET_PATH), "ct_sha256": _sha(CT_PATH), "segmentation_sha256": _sha(SEG_PATH), "evidence_sha256": _sha(EVIDENCE_PATH)}, "ct_window": {"center_hu": 40.0, "width_hu": 400.0, "display_min_hu": CT_VMIN, "display_max_hu": CT_VMAX}, "pet_display_range_suv": [PET_VMIN, PET_VMAX], "pet_display_note": "PET values above 10 SUV are clipped for display only; annotated SUVmax/SUVmean remain true evidence values.", "crop_size_mm": [100.0, 100.0], "orientation_convention": "RAS anatomical labels derived from affine; axial R/L-P/A, coronal R/L-I/S, sagittal P/A-I/S", "generated_panels": panel_hashes, "runtime_seconds": time.perf_counter() - start}
    (OUT_DIR / "panel_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


if __name__ == "__main__":
    render()
