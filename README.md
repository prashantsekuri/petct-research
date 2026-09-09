# PET/CT Research

Research tooling for inspecting and analysing whole-body PET/CT DICOM studies.
This project is not a clinical diagnostic system.

## Data safety

- Keep all source patient data outside this repository.
- Treat original DICOM studies as read-only: never modify, rename, delete,
  anonymize, or write into their source directories.
- Never upload patient data.
- Store future derived artifacts only under the local, Git-ignored `output/`
  directory.
- Export DICOM metadata only through an explicit tag allowlist. Never dump all
  metadata, because it can contain protected health information (PHI).

Git ignore rules for common medical-image names are only a secondary safeguard.
DICOM files can have no filename extension, so those rules cannot guarantee that
patient data will not be committed. The controlling rule is that source patient
data must remain outside the repository.

## Development setup

Use the Python 3.12 `totalseg_env` virtual environment located beside this
repository. It contains both the project dependencies and the isolated medical
imaging model stack:

```bash
conda activate ../totalseg_env
python -m pip install -e ".[dev]"
python -m pytest
```

The workspace interpreter is configured as `../totalseg_env/bin/python`.

The intended system design is documented in
[`docs/architecture.md`](docs/architecture.md).

## Read-only DICOM inventory

Inventory a source directory located outside this repository:

```bash
python -m petct.inventory /path/to/dicom
```

The command recursively reads only explicitly allowlisted metadata using strict
DICOM parsing and does not read pixel data. It groups objects by study and
series, counts instances, and reports whether rows, columns, slice thickness,
or pixel spacing vary within a series. Non-DICOM and unreadable files are
skipped without printing their names.

The inventory command performs no writes. It provides metadata inspection and
bounded pixel probing, but does not produce derived data.

Inspect allowlisted PT and real-world-value mapping metadata needed for a future
SUV validation step:

```bash
python -m petct.inventory /path/to/dicom --pet-details
```

This mode reports metadata presence and consistency only. It does not calculate
SUV or establish that existing SUV values are correct.

Probe representative stored pixel values from quantitative `PT`/`BQML` series:

```bash
python -m petct.inventory /path/to/dicom --pet-pixel-probe
```

The probe decodes only selected instances, excludes values outside the RWV
mapping range from quantitative probes, and compares a candidate BQML-to-SUVbw
transformation with the RWV SUVbw factor. It does not load a complete volume,
produce NIfTI data, or claim full SUV validation.

## Derived PET and CT NIfTI volumes

Create the two approved derived volumes and allowlisted provenance under the
Git-ignored `output/nifti/` directory:

```bash
python -m petct.convert /path/to/dicom
```

The converter reads the original DICOM study without modifying it. It validates
slice geometry, creates a candidate PET SUVbw volume and a WB CECT HU volume,
canonicalizes each to RAS using permutation/flipping only, and verifies the
written NIfTI geometry. It performs no registration, resampling, segmentation,
or AI inference. Existing derived files require an explicit `--overwrite`.
