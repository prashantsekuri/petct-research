"""Native-CT serial-slice review stacks for selected GLOW-FDG candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "output/evidence/glow_fdg_lesions_ct.json"
CT_PATH = ROOT / "output/nifti/CT_WB_CECT.nii.gz"
PET_PATH = ROOT / "output/nifti/PET_SUVbw.nii.gz"
SEG_PATH = ROOT / "output/segmentation/glow_fdg/prediction/study.nii.gz"
ANATOMY_DIR = ROOT / "output/segmentation/totalsegmentator_fast"
OUT = ROOT / "output/evidence/native_ct_stacks"
TARGETS = ("GLOW_FDG_002", "GLOW_FDG_003", "GLOW_FDG_004", "GLOW_FDG_005", "GLOW_FDG_006")
LEVELS = (2.5, 4.0, 6.0)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _candidate_structures(record: dict[str, object]) -> list[str]:
    names = [x["structure"] for x in record["anatomy"]["direct_overlap"]]
    names.extend(x["structure"] for x in record["anatomy"]["nearest_structures"])
    keep = ("artery", "vein", "aorta", "carotid", "subclavian", "thyroid", "clavicula", "iliac")
    return list(dict.fromkeys(x for x in names if any(k in x.lower() for k in keep)))


def _crop(center: int, size: int, half: int) -> slice:
    width = min(size, 2 * half + 1)
    start = max(0, min(center - half, size - width))
    return slice(start, start + width)


def _plane_world_grid(affine: np.ndarray, crop: tuple[slice, slice, slice], plane: str, index: int) -> tuple[np.ndarray, tuple[float, float]]:
    if plane == "axial":
        xs = np.arange(crop[0].start, crop[0].stop); ys = np.arange(crop[1].start, crop[1].stop)
        xx, yy = np.meshgrid(xs, ys, indexing="xy"); ijk = np.stack([xx, yy, np.full_like(xx, index)], -1); spacing = (float(np.linalg.norm(affine[:3, 1])), float(np.linalg.norm(affine[:3, 0])))
    elif plane == "coronal":
        xs = np.arange(crop[0].start, crop[0].stop); zs = np.arange(crop[2].start, crop[2].stop)
        xx, zz = np.meshgrid(xs, zs, indexing="xy"); ijk = np.stack([xx, np.full_like(xx, index), zz], -1); spacing = (float(np.linalg.norm(affine[:3, 0])), float(np.linalg.norm(affine[:3, 2])))
    else:
        ys = np.arange(crop[1].start, crop[1].stop); zs = np.arange(crop[2].start, crop[2].stop)
        yy, zz = np.meshgrid(ys, zs, indexing="xy"); ijk = np.stack([np.full_like(yy, index), yy, zz], -1); spacing = (float(np.linalg.norm(affine[:3, 1])), float(np.linalg.norm(affine[:3, 2])))
    return nib.affines.apply_affine(affine, ijk), spacing


def _display_plane(data: np.ndarray, crop: tuple[slice, slice, slice], plane: str, index: int) -> np.ndarray:
    if plane == "axial": return data[crop[0], crop[1], index].T
    if plane == "coronal": return data[crop[0], index, crop[2]].T
    return data[index, crop[1], crop[2]].T


def run() -> dict[str, object]:
    evidence = json.loads(EVIDENCE.read_text()); records = {x["candidate_id"]: x for x in evidence["lesions"]}
    ct_img, pet_img, seg_img = nib.load(CT_PATH), nib.load(PET_PATH), nib.load(SEG_PATH)
    if ct_img.shape != (512, 512, 716): raise RuntimeError("unexpected native CT geometry")
    if not np.allclose(ct_img.affine[:3, :3], np.diag(np.diag(ct_img.affine[:3, :3])), atol=1e-4): raise RuntimeError("native CT affine is not axis-aligned for deterministic stack rendering")
    ct, pet, seg = np.asanyarray(ct_img.dataobj, dtype=np.float32), np.asanyarray(pet_img.dataobj, dtype=np.float32), np.asanyarray(seg_img.dataobj) != 0
    pet_inv, ct_affine = np.linalg.inv(pet_img.affine), np.asarray(ct_img.affine, dtype=float)
    labels, _ = ndimage.label(seg, np.ones((3, 3, 3), dtype=np.uint8)); components = []
    for cid, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None: continue
        coords = np.argwhere(labels[slc] == cid) + np.asarray([s.start for s in slc]); world = nib.affines.apply_affine(pet_img.affine, coords); components.append((cid, len(coords), world.mean(0)))
    components.sort(key=lambda x: (-x[1], tuple(float(v) for v in x[2])))
    spacing = np.asarray(nib.affines.voxel_sizes(ct_affine)); inv_ct = np.linalg.inv(ct_affine); OUT.mkdir(parents=True, exist_ok=True); all_results = []
    for record in (records[c] for c in TARGETS):
        cid = record["candidate_id"]; centroid = np.asarray(record["mask"]["centroid_ras_mm"], dtype=float); native = nib.affines.apply_affine(inv_ct, centroid); center = np.rint(native).astype(int); half = np.ceil(40 / spacing).astype(int)
        crops = ( _crop(center[0], ct.shape[0], half[0]), _crop(center[1], ct.shape[1], half[1]), _crop(center[2], ct.shape[2], half[2]) )
        axial_indices = list(range(max(0, center[2] - int(np.ceil(12.5 / spacing[2]))), min(ct.shape[2], center[2] + int(np.ceil(12.5 / spacing[2])) + 1)))
        coronal_indices = list(range(max(0, center[1] - 2), min(ct.shape[1], center[1] + 3))); sagittal_indices = list(range(max(0, center[0] - 2), min(ct.shape[0], center[0] + 3)))
        anatomy_names = _candidate_structures(record); anatomy_images = {name: np.asanyarray(nib.load(ANATOMY_DIR / f"{name}.nii.gz").dataobj) != 0 for name in anatomy_names if (ANATOMY_DIR / f"{name}.nii.gz").is_file()}
        fig = plt.figure(figsize=(20, 13)); grid = fig.add_gridspec(3, 11, hspace=.24, wspace=.03)
        plane_specs = [("axial", axial_indices, 0), ("coronal", coronal_indices, 1), ("sagittal", sagittal_indices, 2)]; provenance_planes = []; rendered_anatomy = set(); component_outline = False
        for row, (plane, indices, axis) in enumerate(plane_specs):
            for col, index in enumerate(indices):
                ax = fig.add_subplot(grid[row, col]); world_grid, plane_spacing = _plane_world_grid(ct_affine, crops, plane, index); ct_plane = _display_plane(ct, crops, plane, index); ax.imshow(ct_plane, cmap="gray", vmin=-160, vmax=240, origin="lower", interpolation="nearest", aspect=plane_spacing[1] / plane_spacing[0])
                pet_idx = nib.affines.apply_affine(pet_inv, world_grid); pet_plane = ndimage.map_coordinates(pet, pet_idx.reshape(-1, 3).T, order=1, mode="constant", cval=np.nan).reshape(world_grid.shape[:2]); present = []
                for level, color in zip(LEVELS, ("yellow", "orange", "red")):
                    finite_pet = pet_plane[np.isfinite(pet_plane)]
                    if len(finite_pet) and float(np.min(finite_pet)) <= level <= float(np.max(finite_pet)):
                        ax.contour(pet_plane, levels=[level], colors=[color], linewidths=.8); present.append(level)
                ax.plot(*({"axial": (center[0] - crops[0].start, center[1] - crops[1].start), "coronal": (center[0] - crops[0].start, center[2] - crops[2].start), "sagittal": (center[1] - crops[1].start, center[2] - crops[2].start)}[plane]), marker="+", color="cyan", ms=8)
                for name, anatomy in anatomy_images.items():
                    mask_plane = _display_plane(anatomy, crops, plane, index)
                    if np.any(mask_plane): ax.contour(mask_plane, levels=[.5], colors="#00bfff", linestyles="--", linewidths=.65); rendered_anatomy.add(name)
                # Display-only GLOW-FDG outline, mapped through world coordinates.
                lesion_idx = nib.affines.apply_affine(pet_inv, world_grid); lesion_plane = ndimage.map_coordinates(seg.astype(np.uint8), lesion_idx.reshape(-1, 3).T, order=0, mode="constant", cval=0).reshape(world_grid.shape[:2])
                if np.any(lesion_plane): ax.contour(lesion_plane, levels=[.5], colors="#39ff14", linestyles="-", linewidths=1.0); component_outline = True
                offset = float(nib.affines.apply_affine(ct_affine, [0, 0, index])[2] - centroid[2]) if plane == "axial" else float(nib.affines.apply_affine(ct_affine, [0, index, 0])[1] - centroid[1]) if plane == "coronal" else float(nib.affines.apply_affine(ct_affine, [index, 0, 0])[0] - centroid[0])
                ax.set_title(f"{offset:+.1f} mm", fontsize=7); ax.set_xticks([]); ax.set_yticks([]); provenance_planes.append({"plane": plane, "index": int(index), "offset_mm": offset, "pet_contours_present": present, "pet_contours_absent": [x for x in LEVELS if x not in present]})
        fig.suptitle(f"{cid} | native CT serial review | PET contours: 2.5 yellow, 4 orange, 6 red | GLOW-FDG solid green | anatomy dashed blue", fontsize=11)
        fig.text(.02, .01, "CT window: 40 HU / 400 HU. PET contours are display-only linear samples in native CT planes. No morphology mask used.", fontsize=8)
        out = OUT / f"{cid}.png"; fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
        all_results.append({"candidate_id": cid, "axial_slice_count": len(axial_indices), "axial_physical_range_mm": [min(x["offset_mm"] for x in provenance_planes if x["plane"] == "axial"), max(x["offset_mm"] for x in provenance_planes if x["plane"] == "axial")], "coronal_slice_count": len(coronal_indices), "coronal_offsets_mm": [x["offset_mm"] for x in provenance_planes if x["plane"] == "coronal"], "sagittal_slice_count": len(sagittal_indices), "sagittal_offsets_mm": [x["offset_mm"] for x in provenance_planes if x["plane"] == "sagittal"], "pet_contours": provenance_planes, "anatomy_structures_rendered": sorted(rendered_anatomy), "glow_fdg_outline_rendered": component_outline, "png_sha256": _sha(out)})
    provenance = {"candidate_ids": list(TARGETS), "source_ct_sha256": _sha(CT_PATH), "source_pet_sha256": _sha(PET_PATH), "source_segmentation_sha256": _sha(SEG_PATH), "ct_window": {"center_hu": 40.0, "width_hu": 400.0}, "pet_contour_levels_suv": list(LEVELS), "pet_overlay_method": "native CT world pixel centers mapped to PET grid; scipy linear interpolation for display only", "ct_display_aspect": "physical spacing ratio per plane", "anatomy_overlay_style": "dashed blue contours; GLOW-FDG outline is solid green", "morphology_masks_read": False, "candidates": all_results}; (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


if __name__ == "__main__":
    argparse.ArgumentParser(description="Render native CT serial review stacks.").parse_args(); print(json.dumps(run(), indent=2))
