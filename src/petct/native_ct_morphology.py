"""Conservative native-CT morphology review for selected GLOW-FDG candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "output/evidence/glow_fdg_lesions_ct.json"
ROI_ROOT = ROOT / "output/evidence/native_ct_rois"
ANATOMY_ROOT = ROOT / "output/segmentation/totalsegmentator_fast"
OUT = ROOT / "output/evidence/native_ct_morphology"
TARGETS = ("GLOW_FDG_002", "GLOW_FDG_003", "GLOW_FDG_004", "GLOW_FDG_005", "GLOW_FDG_006")
SEARCH_MM = 15.0
HU_LOW, HU_HIGH = -30.0, 150.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".morphology.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def measure_structure(mask: np.ndarray, ct_image: np.ndarray, affine: np.ndarray) -> dict[str, object]:
    """Measure a native-grid binary structure; mask acquisition is independent."""
    spacing = np.asarray(nib.affines.voxel_sizes(affine), dtype=float)
    coords = np.argwhere(mask)
    if len(coords) == 0: return {}
    values = ct_image[tuple(coords.T)].astype(float)
    world = nib.affines.apply_affine(affine, coords)
    centroid = world.mean(0)
    centered = world - centroid
    pca = {"long_axis_mm": None, "short_axis_mm": None, "third_axis_extent_mm": None, "long_to_short_axis_ratio": None, "principal_axis_extents_mm": None, "null_reason": None}
    if len(coords) >= 4 and np.linalg.matrix_rank(centered) >= 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vt.T
        extents = projected.max(0) - projected.min(0)
        order = np.argsort(extents)[::-1]; extents = extents[order]
        pca.update(long_axis_mm=float(extents[0]), short_axis_mm=float(extents[1]), third_axis_extent_mm=float(extents[2]), long_to_short_axis_ratio=float(extents[0] / extents[1]) if extents[1] > 0 else None, principal_axis_extents_mm=[float(x) for x in extents])
    else:
        pca["null_reason"] = "too_few_or_coplanar_voxels_for_3D_PCA"
    finite = values[np.isfinite(values)]
    stats = {"min": None, "max": None, "mean": None, "median": None, "std": None, "p05": None, "p95": None}
    if len(finite): stats = {"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite)), "median": float(np.median(finite)), "std": float(np.std(finite)), "p05": float(np.percentile(finite, 5)), "p95": float(np.percentile(finite, 95))}
    inner = ndimage.binary_erosion(mask)
    shell = mask & ~inner
    inner_values, shell_values = ct_image[inner], ct_image[shell]
    core = {"status": "indeterminate", "inner_mean_hu": None, "inner_median_hu": None, "peripheral_mean_hu": None, "peripheral_median_hu": None, "difference_hu": None}
    if np.count_nonzero(inner) and np.count_nonzero(shell):
        core.update(status="present" if float(np.mean(inner_values)) < float(np.mean(shell_values)) else "absent", inner_mean_hu=float(np.mean(inner_values)), inner_median_hu=float(np.median(inner_values)), peripheral_mean_hu=float(np.mean(shell_values)), peripheral_median_hu=float(np.median(shell_values)), difference_hu=float(np.mean(inner_values) - np.mean(shell_values)))
    volume_ml = float(len(coords) * np.prod(spacing) / 1000.0)
    return {"voxel_count": int(len(coords)), "volume_ml": volume_ml, "centroid_ras_mm": [float(x) for x in centroid], "ct_hu": stats, "heterogeneity_coefficient_of_variation": float(stats["std"] / abs(stats["mean"])) if stats["mean"] not in (None, 0) else None, "fraction_below_0_hu": float(np.mean(finite < 0)) if len(finite) else None, "fraction_below_20_hu": float(np.mean(finite < 20)) if len(finite) else None, "fraction_above_100_hu": float(np.mean(finite > 100)) if len(finite) else None, "pca_extents": pca, "shape_descriptors": {"sphericity": None, "compactness": None, "elongation": None, "flatness": None, "surface_area_to_volume": None, "null_reason": "not estimated by this conservative voxel geometry method"}, "boundary_descriptors": {"edge_gradient_magnitude": None, "boundary_contrast_hu": None, "null_reason": "not estimated without a stable local boundary model"}, "central_low_attenuation": core}


def _render(cid: str, ct: np.ndarray, mask: np.ndarray, centroid: np.ndarray, affine: np.ndarray, status: str, metrics: dict[str, object], out: Path) -> None:
    spacing = np.asarray(nib.affines.voxel_sizes(affine)); center = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), centroid)).astype(int)
    full = tuple(slice(max(0, center[i] - int(np.ceil(40 / spacing[i]))), min(ct.shape[i], center[i] + int(np.ceil(40 / spacing[i])) + 1)) for i in range(3))
    tight = tuple(slice(max(0, center[i] - int(np.ceil(15 / spacing[i]))), min(ct.shape[i], center[i] + int(np.ceil(15 / spacing[i])) + 1)) for i in range(3))
    fig, ax = plt.subplots(2, 3, figsize=(12, 8)); fig.subplots_adjust(top=.84, wspace=.05, hspace=.08)
    for row, crop in enumerate((full, tight)):
        data, m = ct[crop], mask[crop]; c = center - np.asarray([s.start for s in crop])
        planes = ((data[:, :, c[2]].T, m[:, :, c[2]].T, "AXIAL", "R/L  P/A"), (data[:, c[1], :].T, m[:, c[1], :].T, "CORONAL", "R/L  I/S"), (data[c[0], :, :].T, m[c[0], :, :].T, "SAGITTAL", "P/A  I/S"))
        for col, (image, overlay, title, labels) in enumerate(planes):
            ax[row, col].imshow(image, cmap="gray", vmin=-160, vmax=240, origin="lower", interpolation="nearest")
            if np.any(overlay): ax[row, col].contour(overlay, levels=[.5], colors="#00ff66", linewidths=1.2)
            ax[row, col].plot(image.shape[1] / 2, image.shape[0] / 2, "+", color="cyan", ms=9)
            ax[row, col].set_title(title if row == 0 else f"{title} TIGHT", fontsize=9); ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
            ax[row, col].text(.02, .03, labels, transform=ax[row, col].transAxes, color="white", fontsize=7, bbox={"facecolor":"black","alpha":.5,"pad":2})
            ax[row, col].plot([.08 * image.shape[1], .08 * image.shape[1] + 10 / spacing[(0,1,1)[col]]], [.1 * image.shape[0]] * 2, color="yellow", lw=3)
    pca = metrics.get("pca_extents", {}) if metrics else {}; axes = f"Long/short/third: {pca.get('long_axis_mm')} / {pca.get('short_axis_mm')} / {pca.get('third_axis_extent_mm')} mm"
    fig.suptitle(f"{cid} | status: {status}\n{axes} | PET centroid RAS: ({', '.join(f'{x:.1f}' for x in centroid)})", fontsize=10)
    fig.savefig(out, dpi=130, facecolor="white"); plt.close(fig)


def run() -> dict[str, object]:
    evidence = json.loads(EVIDENCE.read_text()); records = {x["candidate_id"]: x for x in evidence["lesions"]}; source_evidence_sha = _sha(EVIDENCE)
    results = []
    for cid in TARGETS:
        roi_path = ROI_ROOT / cid / "ct_roi.nii.gz"; roi_meta = json.loads((ROI_ROOT / cid / "roi_metadata.json").read_text())
        img = nib.load(roi_path); ct = np.asanyarray(img.dataobj, dtype=np.float32); affine = np.asarray(img.affine, dtype=float); spacing = np.asarray(nib.affines.voxel_sizes(affine), dtype=float)
        centroid = np.asarray(records[cid]["mask"]["centroid_ras_mm"], dtype=float); local_centroid = nib.affines.apply_affine(np.linalg.inv(affine), centroid)
        grid = np.indices(ct.shape, dtype=float).reshape(3, -1).T; distances = np.linalg.norm((grid - local_centroid) * spacing, axis=1).reshape(ct.shape)
        exploratory = (ct >= HU_LOW) & (ct <= HU_HIGH) & (distances <= SEARCH_MM)
        labels, count = ndimage.label(exploratory, np.ones((3, 3, 3), dtype=np.uint8)); components = []
        centroid_voxel = np.rint(local_centroid).astype(int)
        for comp in range(1, count + 1):
            cmask = labels == comp; coords = np.argwhere(cmask); world = nib.affines.apply_affine(affine, coords); d = np.linalg.norm(world - centroid, axis=1); components.append({"id": comp, "mask": cmask, "voxel_count": len(coords), "min_distance_mm": float(np.min(d)), "centroid_distance_mm": float(np.linalg.norm(world.mean(0) - centroid)), "touches_search_boundary": bool(np.any(np.isclose(np.max(d), SEARCH_MM, atol=float(np.max(spacing) * 1.5)))), "contains_pet_centroid": bool(np.all(centroid_voxel >= 0) and np.all(centroid_voxel < ct.shape) and cmask[tuple(centroid_voxel)])})
        containing = [x for x in components if x["contains_pet_centroid"] and x["voxel_count"] >= 3]
        status, reason, selected = "no_clear_discrete_correlate", "no_plausible_component", None
        if len(containing) == 1:
            status, reason, selected = "discrete_correlate_found", "contains_pet_centroid", containing[0]
        elif not containing:
            plausible = [x for x in components if x["voxel_count"] >= 3 and not x["touches_search_boundary"]]
            plausible.sort(key=lambda x: (x["min_distance_mm"], -x["voxel_count"], x["id"]))
            if len(plausible) == 1 or (len(plausible) > 1 and plausible[1]["min_distance_mm"] - plausible[0]["min_distance_mm"] > 3.0):
                status, reason, selected = "discrete_correlate_found", "unique_nearest_component", plausible[0]
            elif len(plausible) > 1:
                status, reason = "ambiguous_multiple_structures", "ambiguous_multiple_nearby_components"
        metrics = measure_structure(selected["mask"], ct, affine) if selected else {}
        if selected:
            coords = np.argwhere(selected["mask"]); selected["min_distance_mm"] = float(np.min(np.linalg.norm(nib.affines.apply_affine(affine, coords) - centroid, axis=1)))
        outdir = OUT / cid; outdir.mkdir(parents=True, exist_ok=True)
        if status == "discrete_correlate_found":
            mask_img = nib.Nifti1Image(selected["mask"].astype(np.uint8), affine); nib.save(mask_img, outdir / "correlate_mask.nii.gz")
        _render(cid, ct, selected["mask"] if selected else np.zeros(ct.shape, bool), centroid, affine, status, metrics, outdir / "review_panel.png")
        result = {"candidate_id": cid, "status": status, "selection_reason": reason, "exploratory_component_count": int(count), "selected_component_distance_from_pet_centroid_mm": selected["min_distance_mm"] if selected else None, "selected_component_contains_pet_centroid": selected["contains_pet_centroid"] if selected else False, "selected_component_touches_search_boundary": selected["touches_search_boundary"] if selected else None, "nearby_vessel_relation": {"evidence_structures": [x for x in records[cid]["anatomy"]["nearest_structures"] if any(t in x["structure"].lower() for t in ("artery", "vein", "aorta"))], "note": "native ROI vessel-mask overlap not inferred from HU threshold"}, "metrics": metrics, "source_roi_sha256": _sha(roi_path), "source_evidence_sha256": source_evidence_sha, "ct_spacing_mm": spacing.tolist(), "search_radius_mm": SEARCH_MM, "exploratory_hu_range": [HU_LOW, HU_HIGH], "selection_method": "26-connected exploratory HU components with centroid/proximity/volume/boundary rules", "morphology_method": "native voxel-center physical geometry and PCA; no RECIST convention", "review_panel_sha256": _sha(outdir / "review_panel.png")}
        if status == "discrete_correlate_found": result["selected_structure_volume_ml"] = metrics["volume_ml"]
        _atomic(outdir / "morphology.json", json.dumps(result, indent=2) + "\n"); results.append(result)
    index = {"source_evidence_sha256": source_evidence_sha, "search_radius_mm": SEARCH_MM, "exploratory_hu_range": [HU_LOW, HU_HIGH], "candidates": [{"candidate_id": x["candidate_id"], "status": x["status"], "review_panel_sha256": x["review_panel_sha256"]} for x in results]}; _atomic(OUT / "index.json", json.dumps(index, indent=2) + "\n"); return index


if __name__ == "__main__":
    argparse.ArgumentParser(description="Analyze selected native CT morphology.").parse_args(); print(json.dumps(run(), indent=2))
