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

Use the pre-created `medical_env` virtual environment located beside this
repository:

```bash
source ../medical_env/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

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

The inventory command performs no writes. PET metadata validation, SUV
validation, volume conversion, segmentation, and measurement functionality are
not implemented.
