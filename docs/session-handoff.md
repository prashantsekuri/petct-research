# Session Handoff

Date: 2026-09-09

## Project purpose and safety boundary

This repository contains research tooling for inspecting and analyzing
whole-body PET/CT studies. It is not a clinical diagnostic system.

The source DICOM study remains outside the repository and is treated as
strictly read-only. The tooling must never modify, rename, delete, anonymize,
or write into the source directory, and patient data must never be uploaded.
Derived files belong only under the Git-ignored `output/` directory. DICOM
filename extensions are not a sufficient protection because valid objects can
have no extension; keeping source data outside the repository is the actual
protection boundary.

Only allowlisted DICOM metadata is read or reported. Do not introduce arbitrary
metadata dumps. Do not put patient identifiers, source paths, or source
filenames into tracked documentation or normal command output.

SUV, lesion size, coordinates, volume, radiomics, and all other measurements
must remain deterministic outputs of imaging code. A future LLM may optionally
interpret an evidence package, but it must not invent or change measurements.

## Repository state

The project uses a modern Python `src` layout and currently declares:

- Python `>=3.11`
- `pydicom>=3.0,<4`
- `numpy>=2.3,<3`
- `nibabel>=5.3,<6`
- `scipy>=1.14,<2`
- `pytest>=8,<10` as a development dependency

Existing commits at handoff:

- `8039caf First commit.`: initial project skeleton and safety documentation
- `f3fa9e9 Added utility to go over dicom ffiles and collect data`: inventory,
  PET metadata, and pixel-probe work
- `79890c8 nifti`: validated PET SUVbw and CT HU NIfTI conversion

Uncommitted repository changes at handoff:

- modified `pyproject.toml` for SciPy
- untracked `.vscode/settings.json`
- untracked `src/petct/seg_audit.py`
- untracked `src/petct/prepare_glow_fdg.py`
- this handoff document

All generated NIfTI, segmentation, and provenance artifacts under `output/`
are ignored by Git and must stay that way.

## Implemented pipeline stages

### 1. Project bootstrap

The repository contains `pyproject.toml`, `README.md`, `AGENTS.md`,
`.gitignore`, `docs/architecture.md`, `src/petct/`, a package smoke test, and an
ignored `output/` tree retained by `output/.gitkeep`.

The ignore rules cover Python environments/build artifacts, common source-data
directory names, common DICOM names/extensions, and derived medical image
formats. These rules are secondary safeguards, not the data-safety boundary.

### 2. Read-only DICOM inventory

`src/petct/inventory.py` recursively scans an external directory with pydicom,
uses normal strict parsing with `force=False`, and skips non-DICOM or unreadable
files. Metadata-only reads use `stop_before_pixels=True` where appropriate.

The inventory groups by `StudyInstanceUID`, then `SeriesInstanceUID`, and emits
a concise series summary from an explicit allowlist. It reports geometry
consistency for `Rows`, `Columns`, `SliceThickness`, and `PixelSpacing` without
rejecting a series solely because one of those fields varies.

Commands:

```bash
python -m petct.inventory /external/read-only/study
python -m petct.inventory /external/read-only/study --pet-details
python -m petct.inventory /external/read-only/study --pet-pixel-probe
```

Normal output excludes patient name, patient ID, birth date, address, accession
number, source paths, and source filenames.

### 3. PET metadata inspection

The PET detail mode identifies PT and RWV objects and collects only the approved
PET quantitation fields. Per-instance observations distinguish `CONSTANT`,
`VARYING`, `PARTIAL`, and `MISSING`. This applies to units, rescale fields, and
timing fields where relevant.

The output includes a conservative `ReadyForSUVValidation` metadata
completeness indicator. It explicitly considers BQML units, usable rescale
values, patient-weight normalization input, injected dose, half-life, injection
timing, `DecayCorrection`, and the `DECY` correction flag. This indicator is not
proof that SUV is clinically correct.

### 4. PET pixel probe and quantitative investigation

The bounded pixel probe selects geometrically representative slices and bed or
acquisition-time groups without loading the complete PET volume. It reports
stored pixel characteristics, DICOM pixel representation, transfer syntax,
rescale terms, BQML values, and RWV mapped ranges.

Important interpretation retained in the implementation:

- DICOM RWVM maps stored pixel values directly and is independent of the
  Modality LUT/rescale transform.
- The standard RWV result is `StoredValue * RWVSlope + RWVIntercept`.
- `BQML * RWVSlope + RWVIntercept` is reported only as a non-standard
  diagnostic where applicable.
- One global stored-value RWV slope is not compatible with all instances when
  instance rescale slopes vary. This is an interpretation issue, not a
  corrupt-study warning.

For this study, the metadata-derived BQML-to-SUVbw factor is
`0.0003324603839118698`, while the linked RWV SUVbw slope is `0.00033246`.
Their percentage difference is approximately `0.000115476%`, well inside the
project's `0.1%` comparison tolerance.

The deterministic candidate transformation used downstream is:

```text
BQML = StoredValue * InstanceRescaleSlope + InstanceRescaleIntercept
SUVbw = BQML * MetadataDerivedBQMLToSUVbwFactor
```

The factor calculation used the documented `DecayCorrection=START` timing
basis. It remains a research transformation and is not a clinical validation.

The RWV declared stored-value range is `0..32765`. The full PET volume contains
550 stored values outside that range, including observed value 32767, across
all 548 source slices. They were counted and preserved through the candidate
transformation; they were not clipped, zeroed, or replaced.

The PET scan contains 548 instances and eight distinct acquisition-time groups.
The probe retains acquisition grouping to describe the multi-bed acquisition.

### 5. Derived PET SUVbw and CT HU NIfTI volumes

`src/petct/convert.py` selects only the `PET Q CLEAR` PT/BQML series and the
`WB CECT` CT series. It rejects inconsistent dimensions, orientation, spacing,
duplicate SOP Instance UIDs, duplicate physical positions, irregular spacing,
and apparent missing slices.

Slices are ordered from `ImagePositionPatient` and
`ImageOrientationPatient`, not filenames. The DICOM row/column affine semantics
are checked explicitly. The affine is converted from DICOM LPS to NIfTI RAS.
Canonicalization uses `nibabel.as_closest_canonical()` with permutation and
flipping only; no interpolation or resampling occurs in this conversion stage.

Command:

```bash
python -m petct.convert /external/read-only/study
```

Use `--overwrite` only when intentionally replacing the four fixed derived
outputs.

Validated PET output:

- path: `output/nifti/PET_SUVbw.nii.gz`
- shape: `(192, 192, 548)`
- spacing: `(3.645833, 3.645833, 3.26)` mm
- orientation: RAS
- voxel-center bounds: `[-348.177063, -348.177063, -1909.864990]` to
  `[348.177088, 348.177088, -126.644995]` mm
- SUVbw minimum/maximum/mean: `0 / 14.0585756 / 0.0642629706`
- finite/non-finite voxels: `20,201,472 / 0`
- SHA-256: `f7bddf68116191759ae45b6d3e4ffa806f2b330d0ab6b0426063bcbc5361261d`

Validated CT output:

- path: `output/nifti/CT_WB_CECT.nii.gz`
- shape: `(512, 512, 716)`
- spacing: `(0.982422, 0.982422, 2.5)` mm
- orientation: RAS
- voxel-center bounds: `[-250.517639, -250.517639, -1911.25]` to
  `[251.5, 251.5, -123.75]` mm
- HU minimum/maximum/mean: `-3024 / 3071 / -1242.0606899`
- finite/non-finite voxels: `187,695,104 / 0`
- SHA-256: `3cc917d3abd8cec71569acde7b32cd30c70dfe8a7e0a3c09bfa46c871ad9053d`

Both original arrays were permuted and flipped during RAS canonicalization.
Every recorded post-write NIfTI verification passed. Technical, allowlisted
provenance is stored in `output/nifti/pet_provenance.json` and
`output/nifti/ct_provenance.json`.

### 6. TotalSegmentator environment and anatomy-mask audit

A separate Python 3.12 conda environment exists at `../totalseg_env/`. It
currently contains:

- Python 3.12.14
- TotalSegmentator 2.18.0
- PyTorch 2.14.0
- nnU-Net v2 2.8.1
- NumPy 2.5.3
- nibabel 5.4.2
- pydicom 3.0.2
- SciPy 1.18.1

`.vscode/settings.json` points VS Code at
`${workspaceFolder}/../totalseg_env/bin/python`. That settings file is currently
untracked.

A fast TotalSegmentator result is present at
`output/segmentation/totalsegmentator_fast/`. The exact inference invocation
and run log are not recorded in the repository, so confirm those separately
before treating the masks as reproducible model output.

`src/petct/seg_audit.py` validates those masks against the CT without writing
files:

```bash
python -m petct.seg_audit
```

Latest audit result:

- total masks: 117
- PASS: 115
- EMPTY: 2
- shape mismatches: 0
- affine mismatches: 0
- empty masks: `kidney_cyst_right.nii.gz` and `prostate.nii.gz`
- largest masks include liver, brain, colon, small bowel, and skull

An empty model output is recorded as `EMPTY`; it is not automatically treated
as a geometry failure or a statement that the anatomy/pathology is absent.

### 7. Lesion-model research and model choice

LesionTracer was reviewed but reserved for a possible later comparison. Its
official implementation is the
[AutoPET III submission repository](https://github.com/MIC-DKFZ/autopet-3-submission),
with the [paper](https://arxiv.org/abs/2409.09478) and
[published weights](https://zenodo.org/records/14007247).

GLOW-FDG was selected as the primary openly available FDG whole-body PET/CT
lesion model because it is newer, trained specifically for FDG data, and uses a
vanilla nnU-Net-compatible input. Relevant resources are the
[GLOW-FDG repository](https://github.com/MIC-DKFZ/GLOW-FDG) and
[model weights](https://huggingface.co/mrokuss/GLOW-FDG).

The reviewed GLOW-FDG input contract is:

- CT in HU as channel 0: `*_0000.nii.gz`
- PET in SUVbw as channel 1: `*_0001.nii.gz`
- both inputs on exactly the same shape, affine, and physical grid
- model planning uses approximately `3 x 2.04 x 2.04` mm spacing and a
  `192 x 192 x 192` patch; no manual conversion to that model spacing was done
- CT/PET registration is assumed before inference

The code repository is Apache-2.0. The published model weights are
CC BY-NC-SA 4.0, so use remains non-commercial and must comply with attribution
and share-alike terms. Confirm the current upstream terms before download or
distribution.

Apple Silicon MPS execution has not yet been validated for GLOW-FDG in this
project. Do not assume that nnU-Net inference will work correctly on MPS merely
because another package accepts a `--device mps` option. CPU remains a possible
fallback, with a substantial runtime cost.

### 8. GLOW-FDG paired-input preparation

`src/petct/prepare_glow_fdg.py` prepares the paired channels using fixed project
paths:

```bash
python -m petct.prepare_glow_fdg
```

Existing outputs require `--overwrite`. The PET grid is authoritative. PET is
copied byte-for-byte without interpolation, clipping, normalization, or header
regeneration. CT is linearly resampled to the PET grid with
`scipy.ndimage.affine_transform` using:

```text
PET voxel -> PET affine -> RAS world -> inverse CT affine -> CT voxel
```

No registration or transform optimization is performed. CT target centers
outside the source CT voxel-center domain receive `-1024 HU`.

Generated model inputs:

- `output/segmentation/glow_fdg/input/study_0000.nii.gz`: CT on PET grid
- `output/segmentation/glow_fdg/input/study_0001.nii.gz`: unchanged PET SUVbw
- `output/segmentation/glow_fdg/input/preprocessing_provenance.json`

Validation findings:

- output shape: `(192, 192, 548)` for both channels
- output spacing: `(3.645833, 3.645833, 3.26)` mm
- orientation: RAS
- CT affine exactly equals the PET affine
- copied PET is byte-identical to the source derived PET
- CT output HU minimum/maximum/mean: `-3024 / 3071 / -1136.7562003`
- both outputs have 20,201,472 finite voxels and zero non-finite voxels
- PET and source CT physical axis-aligned bounds overlap
- 10,436,112 PET voxel centers, or `51.660156%`, lie inside the source CT
  voxel-center domain
- 9,765,360 target CT voxels, or `48.339844%`, are outside that domain and were
  filled with `-1024 HU`
- all eight PET-grid corners lie outside the narrower transaxial CT FOV; the
  deterministic center point lies inside
- all post-write shape, affine, spacing, orientation, statistics, hash, and
  outside-fill checks passed

The large fill percentage is explained by the PET grid being much wider than
the CT transaxial FOV. It is recorded rather than hidden. The source CT also
contains values as low as `-3024 HU`; these were preserved and not clipped.

Coarse descriptive alignment extents on the common PET grid:

| Threshold | Voxels | Physical voxel-center bounds (mm) |
| --- | ---: | --- |
| PET SUVbw > 0 | 9,367,496 | `[-344.531230, -348.177063, -1909.864990]` to `[348.177088, 344.531255, -126.644995]` |
| PET SUVbw > 0.1 | 1,763,869 | `[-293.489564, -282.552064, -1909.864990]` to `[337.239589, 264.322924, -126.644995]` |
| PET SUVbw > 0.5 | 888,441 | `[-216.927066, -264.322898, -1909.864990]` to `[220.572925, 253.385424, -126.644995]` |
| CT HU > -500 | 1,800,523 | `[-246.093732, -249.739565, -1909.864990]` to `[235.156258, 216.927091, -126.644995]` |

All three PET threshold bounding boxes overlap the CT tissue bounding box.
These thresholds are descriptive only. They were not used to crop, register,
normalize, or classify alignment as passing or failing.

Output hashes:

- CT channel SHA-256: `23f255df27fda5e9463ccb877dac1aabbabc1397abef000fc0f09c6f2fe10f4a`
- PET channel SHA-256: `f7bddf68116191759ae45b6d3e4ffa806f2b330d0ab6b0426063bcbc5361261d`
- latest preprocessing provenance SHA-256:
  `2827bb5e2c106efe223ba722a96d8bb72bc19bd496324c7d90ce4227765e8041`

The CT and PET hashes were stable across repeated preparation runs. The
provenance hash changes when its creation timestamp changes.

## Environment nuances

The current shell resolves `python` to `../medical_env/bin/python`, which is
Python 3.14.5 with NumPy 2.5.3, nibabel 5.4.2, pydicom 3.0.2, and SciPy 1.18.1.
VS Code instead defaults to the Python 3.12 `totalseg_env`. Always check
`command -v python` and `python --version` before running a pipeline command.

The intended `../glow_fdg_env/` does not exist yet. `../glow_fdg_home/` does
exist and contains Hugging Face configuration plus a small GLOW-FDG
`dataset.json`, but no model checkpoint files were found at this handoff. It
also contains local authentication files. Never copy, print, or commit those
credentials.

No GLOW-FDG package, inference environment, checkpoint, or lesion output was
created during the preprocessing step.

## Verification completed at handoff

```text
python -m pytest
1 passed

python -m pip check
No broken requirements found

python -m petct.seg_audit
117 masks: 115 PASS, 2 EMPTY, 0 geometry mismatches

python -m petct.prepare_glow_fdg --overwrite
All paired-input post-write checks passed
```

The pip invocation emitted a local cache-permission warning and disabled its
cache. Dependency validation still completed successfully.

## Pending work

### Immediate repository housekeeping

1. Review and commit the current uncommitted code and documentation.
2. Decide whether `.vscode/settings.json` should be committed or remain a local
   workspace preference.
3. Update `README.md` and `docs/architecture.md`. They currently stop at the
   DICOM-to-NIfTI stage and still say segmentation/resampling is not
   implemented.
4. Consider recording future model commands, package locks, logs, and model
   identifiers in non-PHI provenance so the anatomy and lesion outputs can be
   reproduced.

### GLOW-FDG environment and weights

1. Re-check the current official GLOW-FDG repository, nnU-Net requirements,
   model layout, license, and available PyTorch wheels immediately before
   installation.
2. Create the isolated Python 3.12 environment at `../glow_fdg_env/`; do not add
   GLOW-FDG or PyTorch to the main project dependencies.
3. Use `../glow_fdg_home/` for model/cache state and keep credentials and model
   weights outside the repository.
4. The previously proposed model stack was Python 3.12, PyTorch 2.14.0,
   nnU-Net v2 2.8.1, and `huggingface_hub` 1.30.0. Treat these as provisional
   until checked against the current upstream model instructions.
5. Download weights only after explicitly approving the license and download.
6. Validate MPS on a controlled first inference or use CPU if MPS is unsupported
   or unstable. Do not silently change devices.

### Lesion inference and validation

1. Visually inspect the paired CT/PET inputs in a local medical-image viewer.
   The numerical checks establish grid consistency but do not prove anatomical
   registration.
2. Run GLOW-FDG only after the dedicated environment and weights are verified.
3. Keep inference output under `output/segmentation/glow_fdg/` and preserve the
   prepared `study_0000.nii.gz` and `study_0001.nii.gz` channel convention.
4. Record the exact model version, checkpoint hashes, command, device, runtime,
   and nnU-Net configuration in provenance.
5. Add a deterministic lesion-mask audit for shape, affine, label values,
   non-finite data, voxel count, and physical bounds before measurements.
6. Do not call lesion segmentation clinically validated based on model output
   alone. Retain visual and quantitative quality-control findings.

### Later deterministic stages

1. Define deterministic lesion components and anatomy-mask relationships.
2. Implement deterministic SUV statistics, lesion volume, dimensions,
   coordinates, and any radiomics with explicit conventions and provenance.
3. Build a non-PHI evidence package containing model versions, input/output
   hashes, geometry, quality checks, and deterministic measurements.
4. Add optional LLM interpretation only after the evidence package exists. The
   LLM must never modify or become the source of quantitative values.
5. Keep the actual LesionTracer integration, if revisited, under a distinct
   `output/segmentation/lesiontracer/` namespace.

## Recommended resume sequence

```bash
git status --short
command -v python
python --version
python -m pytest
python -m petct.seg_audit
```

Then review this handoff, inspect the paired GLOW-FDG provenance, re-verify the
official model installation contract, and request explicit approval before
creating the model environment or downloading weights.
