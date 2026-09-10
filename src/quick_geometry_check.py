# import nibabel as nib
# import numpy as np

# pet = nib.load("output/segmentation/glow_fdg/input/study_0001.nii.gz")
# seg = nib.load("output/segmentation/glow_fdg/prediction_fold0/study.nii.gz")

# data = np.asarray(seg.dataobj)

# print("PET shape:", pet.shape)
# print("SEG shape:", seg.shape)
# print("Affine match:", np.allclose(pet.affine, seg.affine, rtol=0, atol=1e-5))
# print("SEG dtype:", data.dtype)
# print("Unique labels:", np.unique(data))
# print("Lesion voxels:", np.count_nonzero(data))
# print("Lesion fraction:", np.count_nonzero(data) / data.size)


# import nibabel as nib
# import numpy as np

# pet = nib.load("output/segmentation/glow_fdg/input/study_0001.nii.gz")
# seg = nib.load("output/segmentation/glow_fdg/prediction/study.nii.gz")

# data = np.asarray(seg.dataobj)

# print("PET shape:", pet.shape)
# print("SEG shape:", seg.shape)
# print("Affine match:", np.allclose(pet.affine, seg.affine, rtol=0, atol=1e-5))
# print("SEG dtype:", data.dtype)
# print("Unique labels:", np.unique(data))
# print("Lesion voxels:", np.count_nonzero(data))
# print("Lesion fraction:", np.count_nonzero(data) / data.size)


import nibabel as nib
import numpy as np
from scipy import ndimage

seg_path = "output/segmentation/glow_fdg/prediction/study.nii.gz"
pet_path = "output/segmentation/glow_fdg/input/study_0001.nii.gz"

seg_img = nib.load(seg_path)
pet_img = nib.load(pet_path)

seg = np.asarray(seg_img.dataobj)
pet = np.asarray(pet_img.dataobj)

structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity
labels, n = ndimage.label(seg > 0, structure=structure)

voxel_volume_ml = abs(np.linalg.det(seg_img.affine[:3, :3])) / 1000.0

print("Total lesion voxels:", int(np.count_nonzero(seg)))
print("Number of connected components:", n)
print()

components = []

for label_id in range(1, n + 1):
    mask = labels == label_id
    count = int(mask.sum())

    coords = np.argwhere(mask)
    vmin = coords.min(axis=0)
    vmax = coords.max(axis=0)

    centroid_voxel = coords.mean(axis=0)
    centroid_world = nib.affines.apply_affine(seg_img.affine, centroid_voxel)

    pet_vals = pet[mask]

    components.append({
        "id": label_id,
        "voxels": count,
        "volume_ml": count * voxel_volume_ml,
        "vmin": vmin,
        "vmax": vmax,
        "centroid_world": centroid_world,
        "suvmax": float(pet_vals.max()),
        "suvmean": float(pet_vals.mean()),
    })

components.sort(key=lambda x: x["voxels"], reverse=True)

for i, c in enumerate(components, 1):
    print(f"Component {i}")
    print(f"  Voxel count: {c['voxels']}")
    print(f"  Approximate volume mL: {c['volume_ml']:.6f}")
    print(
        "  Voxel bounding box: "
        f"minimum={tuple(c['vmin'])}, maximum={tuple(c['vmax'])}"
    )
    print(
        "  Physical centroid RAS mm: "
        f"({c['centroid_world'][0]:.6f}, "
        f"{c['centroid_world'][1]:.6f}, "
        f"{c['centroid_world'][2]:.6f})"
    )
    print(f"  SUVmax: {c['suvmax']:.8f}")
    print(f"  SUVmean: {c['suvmean']:.8f}")
    print()
