"""Constrained, evidence-only PET/CT report generation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "output/evidence/glow_fdg_lesions_ct.json"
REPORT_DIR = ROOT / "output/reports"
JSON_PATH = REPORT_DIR / "ai_petct_report.json"
MD_PATH = REPORT_DIR / "ai_petct_report.md"
PROMPT_VERSION = "petct-report-v1"
FORBIDDEN_TERMS = re.compile(r"\b(metastasis|metastatic|malignant|malignancy|lymphoma|tuberculosis|tb|infectious node|benign|physiological uptake|pathological uptake)\b", re.I)


class ReportBackend(Protocol):
    name: str

    def generate_report(self, evidence_payload: dict[str, object]) -> dict[str, object]:
        ...


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _payload(evidence: dict[str, object]) -> dict[str, object]:
    # Evidence is already PHI-free; select only fields permitted into prompting.
    return {"model": evidence.get("model", {}), "reference_grid": evidence.get("reference_grid", {}), "analysis": evidence.get("analysis", {}), "lesions": evidence["lesions"]}


class TemplateBackend:
    name = "template"

    def generate_report(self, evidence_payload: dict[str, object]) -> dict[str, object]:
        lesions = evidence_payload["lesions"]
        ids = [item["candidate_id"] for item in lesions]
        findings = []
        for item in lesions:
            cid = item["candidate_id"]
            overlap = item["anatomy"]["direct_overlap"]
            nearest = item["anatomy"]["nearest_structures"][:3]
            location = ", ".join(x["structure"] for x in overlap) if overlap else ", ".join(x["structure"] for x in nearest) or "no named nearby structure"
            uncertainties = ["Anatomical localization is descriptive and may involve multiple overlapping structures."]
            findings.append({"candidate_id": cid, "description": f"Candidate {cid} is a region of increased FDG uptake localized near {location}.", "evidence_fields": ["pet.suvmax", "pet.suvmean", "mask.volume_ml", "ct.intensity_hu.mean", "anatomy.direct_overlap"], "uncertainties": uncertainties})
        top = sorted(lesions, key=lambda x: x["pet"]["suvmax"], reverse=True)[:3]
        return {"summary": {"candidate_count": len(lesions), "text": "The pipeline identified the recorded number of GLOW-FDG ensemble candidate regions. This is a research second-reading summary.", "candidate_ids": []}, "findings": findings, "distribution": {"text": "Candidate regions are distributed across the anatomical structures listed in the evidence records, including pelvic, bowel-associated, and upper thoracic or lower-neck regions where supported.", "candidate_ids": ids}, "impression": [{"text": "The ensemble produced candidate regions requiring visual review against the deterministic PET, CT, and anatomy evidence.", "candidate_ids": ids}, {"text": "The highest measured uptake candidates are listed by identifier in the structured evidence.", "candidate_ids": [x["candidate_id"] for x in top]}], "limitations": ["This is a research AI second-reading workflow, not a clinical diagnostic report.", "Anatomical overlap and proximity are geometry-based evidence and do not establish tissue identity or disease classification.", "No radiologist report was used."]}


def _inject_numbers(report: dict[str, object], evidence: dict[str, object]) -> None:
    source = {x["candidate_id"]: x for x in evidence["lesions"]}
    for finding in report["findings"]:
        item = source[finding["candidate_id"]]
        finding["numeric_evidence"] = {"suvmax": item["pet"]["suvmax"], "suvmean": item["pet"]["suvmean"], "volume_ml": item["mask"]["volume_ml"]}


def _validate(report: dict[str, object], evidence: dict[str, object]) -> None:
    ids = [x["candidate_id"] for x in evidence["lesions"]]
    findings = report.get("findings")
    if not isinstance(findings, list) or len(findings) != len(ids):
        raise ValueError("finding count does not match source evidence")
    found = [x.get("candidate_id") for x in findings]
    if found != ids or len(set(found)) != len(ids):
        raise ValueError("candidate IDs are missing, reordered, duplicated, or unknown")
    for finding in findings:
        if finding["candidate_id"] not in finding.get("description", ""):
            raise ValueError("finding description lacks candidate citation")
    for section in (report.get("distribution", {}),):
        if section.get("text") and not section.get("candidate_ids"):
            raise ValueError("distribution statement lacks candidate citations")
    for item in report.get("impression", []):
        if item.get("text") and not item.get("candidate_ids"):
            raise ValueError("impression statement lacks candidate citations")
    prose = [report.get("summary", {}).get("text", ""), report.get("distribution", {}).get("text", "")]
    prose.extend(x.get("description", "") for x in report.get("findings", []))
    prose.extend(x for x in report.get("limitations", []))
    prose.extend(x.get("text", "") for x in report.get("impression", []))
    text = "\n".join(prose)
    if FORBIDDEN_TERMS.search(text):
        raise ValueError("forbidden diagnostic terminology detected")
    stripped = re.sub(r"GLOW_FDG_\d+", "", text)
    if re.search(r"(?<![A-Za-z_])[+-]?\d+(?:\.\d+)?", stripped):
        raise ValueError("unvalidated numeric token found in generated text")


def _render_markdown(report: dict[str, object]) -> str:
    lines = ["# AI PET/CT Research Draft Report", "", "## TECHNIQUE / PIPELINE", "Research AI second-reading workflow using the GLOW-FDG five-fold ensemble. PET and CT were spatially harmonized on the PET reference grid. Quantitative measurements are deterministic and sourced from the evidence pipeline.", "", "## FINDINGS"]
    for finding in report["findings"]:
        n = finding["numeric_evidence"]
        lines += [f"### {finding['candidate_id']}", finding["description"], f"- SUVmax: {n['suvmax']:.12g} [{finding['candidate_id']}]", f"- SUVmean: {n['suvmean']:.12g} [{finding['candidate_id']}]"]
        source = report["_source_by_id"][finding["candidate_id"]]
        dims = source["ct"]["dimensions_mm"]
        lines += [f"- Predicted mask volume: {n['volume_ml']:.12g} mL [{finding['candidate_id']}]"]
        lines += [f"- Approximate dimensions (bbox): {dims['bbox_extent_x']:.6g} x {dims['bbox_extent_y']:.6g} x {dims['bbox_extent_z']:.6g} mm [{finding['candidate_id']}]"]
        lines += [f"- CT HU mean: {source['ct']['intensity_hu']['mean']:.12g}; median: {source['ct']['intensity_hu']['median']:.12g} [{finding['candidate_id']}]"]
        overlap = source["anatomy"]["direct_overlap"]
        lines += ["- Direct overlap: " + (", ".join(f"{x['structure']} ({x['percent']:.3g}%)" for x in overlap) if overlap else "none") + f" [{finding['candidate_id']}]"]
        lines += ["- Nearest structures: " + (", ".join(f"{x['structure']} ({x['distance_mm']:.3g} mm)" for x in source["anatomy"]["nearest_structures"][:3]) or "none") + f" [{finding['candidate_id']}]"]
        vertebra = source["anatomy"]["nearest_vertebra"]
        lines += ["- Nearest vertebra: " + (f"{vertebra['structure']} ({vertebra['lesion_to_mask_distance_mm']:.3g} mm)" if vertebra["structure"] else "none") + f" [{finding['candidate_id']}]"]
        lines.append("")
    lines += ["## DISTRIBUTION", report["distribution"]["text"] + " [" + ", ".join(report["distribution"]["candidate_ids"]) + "]", "", "## UNCERTAINTIES"]
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines += ["", "## IMPRESSION"]
    for item in report["impression"]:
        lines.append(f"- {item['text']} [{', '.join(item['candidate_ids'])}]")
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".report.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def run() -> dict[str, object]:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)
    payload = _payload(evidence)
    backend = TemplateBackend()
    generated = backend.generate_report(payload)
    _inject_numbers(generated, evidence)
    _validate(generated, evidence)
    generated["_source_by_id"] = {x["candidate_id"]: x for x in evidence["lesions"]}
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {"report_version": 1, "source_evidence_sha256": _sha_bytes(evidence_bytes), "generator": {"type": "template", "model": backend.name, "prompt_version": PROMPT_VERSION, "generated_at_utc": generated_at, "prompt_sha256": _sha_bytes(_canonical(payload))}, **generated}
    validated_copy = copy.deepcopy(report)
    validated_copy["generator"].pop("validated_report_payload_sha256", None)
    report["generator"]["validated_report_payload_sha256"] = _sha_bytes(_canonical(validated_copy))
    report.pop("_source_by_id")
    _atomic_text(JSON_PATH, json.dumps(report, indent=2) + "\n")
    render_source = dict(report); render_source["_source_by_id"] = {x["candidate_id"]: x for x in evidence["lesions"]}
    _atomic_text(MD_PATH, _render_markdown(render_source))
    return report


def main() -> int:
    argparse.ArgumentParser(description="Generate constrained evidence-only PET/CT report.").parse_args()
    report = run()
    print(json.dumps({"backend": report["generator"]["model"], "candidate_count": report["summary"]["candidate_count"], "validation": "PASS", "source_evidence_sha256": report["source_evidence_sha256"], "prompt_sha256": report["generator"]["prompt_sha256"], "validated_payload_sha256": report["generator"]["validated_report_payload_sha256"], "json_sha256": _sha_file(JSON_PATH), "markdown_sha256": _sha_file(MD_PATH), "json_size": JSON_PATH.stat().st_size, "markdown_size": MD_PATH.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
