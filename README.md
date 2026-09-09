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

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

The intended system design is documented in
[`docs/architecture.md`](docs/architecture.md). DICOM processing is not yet
implemented.
