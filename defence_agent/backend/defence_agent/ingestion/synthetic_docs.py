from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from docx import Document as DocxDocument
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from defence_agent.config import get_settings


@dataclass(frozen=True)
class SyntheticDocSpec:
    doc_id: str
    filename: str
    title: str
    doc_family: str
    version: str
    status: str
    language: str
    access_level: str
    owner: str
    effective_date: str
    review_due: str
    source_type: str
    sections: tuple[tuple[str, int, str], ...]

    @property
    def markdown_filename(self) -> str:
        return f"{self.doc_id}.md"


SPECS: tuple[SyntheticDocSpec, ...] = (
    SyntheticDocSpec(
        doc_id="PB-SOP-2025",
        filename="PB-SOP-2025.docx",
        title="Planning Brief Approval SOP",
        doc_family="planning_brief_approval",
        version="2025.1",
        status="approved",
        language="en",
        access_level="public_internal",
        owner="Planning Policy Directorate",
        effective_date="2025-02-01",
        review_due="2026-03-15",
        source_type="docx",
        sections=(
            (
                "Current Approval Process",
                1,
                "The current approved planning brief approval process requires intake and purpose classification, evidence pack review, planning brief drafting, section chief review within 2 business days, legal or policy review when the brief includes external distribution, restricted annexes, or cross-departmental commitments, director approval before circulation, and archival of the evidence and decision log.",
            ),
            (
                "Urgent Exception Rule",
                2,
                "Urgency may compress the order of sign-offs, but it cannot skip evidence review. The rationale must be recorded, and retrospective director confirmation is required within 1 business day.",
            ),
            (
                "Evidence Pack Cross-Reference",
                2,
                "Before review, the required evidence pack must satisfy PB-CHK-2025.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="PB-SOP-2024",
        filename="PB-SOP-2024.docx",
        title="Planning Brief Approval SOP",
        doc_family="planning_brief_approval",
        version="2024.3",
        status="superseded",
        language="en",
        access_level="public_internal",
        owner="Planning Policy Directorate",
        effective_date="2024-01-10",
        review_due="2025-01-10",
        source_type="docx",
        sections=(
            (
                "Superseded Review Process",
                1,
                "The superseded 2024 planning brief process allowed section chief review within 3 business days. Legal and policy review was required only for public release.",
            ),
            (
                "Urgent Provisional Circulation",
                2,
                "Urgent provisional circulation was allowed after duty officer approval. Evidence review was recommended but not mandatory before urgent provisional circulation.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="PB-SOP-2026-DRAFT",
        filename="PB-SOP-2026-DRAFT.docx",
        title="Planning Brief Approval SOP",
        doc_family="planning_brief_approval",
        version="2026.0-draft",
        status="draft",
        language="en",
        access_level="public_internal",
        owner="Planning Policy Directorate",
        effective_date="2026-04-01",
        review_due="2026-12-31",
        source_type="docx",
        sections=(
            (
                "Draft Pilot Language",
                1,
                "Draft only. This pilot proposes self-approval for low-risk briefs after automated checklist completion. This language is not approved guidance and must not be used for current approved procedure questions.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="PB-CHK-2025",
        filename="PB-CHK-2025.pdf",
        title="Planning Brief Evidence Checklist",
        doc_family="planning_brief_checklist",
        version="2025.2",
        status="approved",
        language="en",
        access_level="public_internal",
        owner="Planning Policy Directorate",
        effective_date="2025-02-01",
        review_due="2026-08-01",
        source_type="pdf",
        sections=(
            (
                "Required Evidence Pack",
                1,
                "Before review, the evidence pack must include a problem statement, policy basis, source list, assumptions, impacted stakeholders, risk rating, options considered, recommendation, citation table, open questions, and decision log stub.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="EC-PROC-2025",
        filename="EC-PROC-2025.pdf",
        title="Emergency Communications Procedure",
        doc_family="emergency_communications",
        version="2025.1",
        status="approved",
        language="en",
        access_level="public_internal",
        owner="Emergency Coordination Office",
        effective_date="2025-06-01",
        review_due="2026-04-01",
        source_type="pdf",
        sections=(
            (
                "Activation And Approval Gates",
                1,
                "The emergency communications procedure covers activation conditions, approval gates, and required evidence. Public dissemination requires citation to the reporting guide.",
            ),
            (
                "Timelines",
                1,
                "The initial acknowledgement is due within 15 minutes, the operational situation update within 45 minutes, the executive summary within 90 minutes, and the final record within 1 business day.",
            ),
            (
                "Required Evidence",
                2,
                "Required evidence includes source event log, responsible desk, time of receipt, confidence level, distribution list, and approving official.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="EC-PROC-2025-FR",
        filename="EC-PROC-2025-FR.pdf",
        title="Procedure de communications d'urgence",
        doc_family="emergency_communications",
        version="2025.1-fr",
        status="approved",
        language="fr",
        access_level="public_internal",
        owner="Bureau de coordination des urgences",
        effective_date="2025-06-01",
        review_due="2026-04-01",
        source_type="pdf",
        sections=(
            (
                "Delais",
                1,
                "La procedure de communications d'urgence exige un accuse de reception initial dans les 15 minutes, une mise a jour operationnelle dans les 45 minutes, un resume executif dans les 90 minutes, et un dossier final dans un jour ouvrable.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="ANNEX-HANDLING-2025",
        filename="ANNEX-HANDLING-2025.pdf",
        title="Restricted Annex Handling Guide",
        doc_family="annex_handling",
        version="2025.1",
        status="approved",
        language="en",
        access_level="restricted",
        owner="Records and Security Office",
        effective_date="2025-07-01",
        review_due="2026-06-30",
        source_type="pdf",
        sections=(
            (
                "External Distribution Control",
                1,
                "Restricted annex handling before external distribution requires access confirmation, lead approval, distribution minimization, annex marking validation, and an evidence log entry. This is synthetic safe content.",
            ),
            (
                "Disclosure Boundary",
                1,
                "Restricted annex handling details must not be shown to users without restricted access and an authorized role.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="LOG-RET-2025",
        filename="LOG-RET-2025.docx",
        title="Evidence and Decision Log Retention Procedure",
        doc_family="evidence_log_retention",
        version="2025.1",
        status="approved",
        language="en",
        access_level="public_internal",
        owner="Records Management Office",
        effective_date="2025-01-15",
        review_due="2026-01-30",
        source_type="docx",
        sections=(
            (
                "Evidence And Decision Logs",
                1,
                "Evidence logs must record source title, source version, section or page, reviewer, and decision timestamp. Decision logs must be retained for 7 years.",
            ),
            (
                "Traceability Requirement",
                1,
                "The log must distinguish facts from assumptions. Each briefing decision must include a trace ID.",
            ),
        ),
    ),
    SyntheticDocSpec(
        doc_id="SCANNED-MANUAL-EXCERPT-1999",
        filename="SCANNED-MANUAL-EXCERPT-1999.pdf",
        title="Legacy Field Manual Excerpt",
        doc_family="legacy_manual",
        version="1999-scan",
        status="superseded",
        language="en",
        access_level="public_internal",
        owner="Archives",
        effective_date="1999-03-01",
        review_due="2000-03-01",
        source_type="scanned_pdf",
        sections=(
            (
                "OCR Simulated Legacy Process",
                1,
                "OCR simulated text: LEG4CY MANUAL EXCERPT. Approval was paper-based and routed by courier. This old process is superseded and should not answer current approved procedure questions.",
            ),
        ),
    ),
)


TRACKER_ROWS: tuple[tuple[str, str, str, str, str, str, str, str, str], ...] = (
    ("PB-SOP-2025", "Planning Brief Approval SOP", "planning_brief_approval", "Planning Policy Directorate", "approved", "2025-02-01", "2026-03-15", "public_internal", "en"),
    ("PB-SOP-2024", "Planning Brief Approval SOP", "planning_brief_approval", "Planning Policy Directorate", "superseded", "2024-01-10", "2025-01-10", "public_internal", "en"),
    ("PB-SOP-2026-DRAFT", "Planning Brief Approval SOP", "planning_brief_approval", "Planning Policy Directorate", "draft", "2026-04-01", "2026-12-31", "public_internal", "en"),
    ("PB-CHK-2025", "Planning Brief Evidence Checklist", "planning_brief_checklist", "Planning Policy Directorate", "approved", "2025-02-01", "2026-08-01", "public_internal", "en"),
    ("EC-PROC-2025", "Emergency Communications Procedure", "emergency_communications", "Emergency Coordination Office", "approved", "2025-06-01", "2026-04-01", "public_internal", "en"),
    ("EC-PROC-2025-FR", "Procedure de communications d'urgence", "emergency_communications", "Emergency Coordination Office", "approved", "2025-06-01", "2026-04-01", "public_internal", "fr"),
    ("ANNEX-HANDLING-2025", "Restricted Annex Handling Guide", "annex_handling", "Records and Security Office", "approved", "2025-07-01", "2026-06-30", "restricted", "en"),
    ("LOG-RET-2025", "Evidence and Decision Log Retention Procedure", "evidence_log_retention", "Records Management Office", "approved", "2025-01-15", "2026-01-30", "public_internal", "en"),
    ("SCANNED-MANUAL-EXCERPT-1999", "Legacy Field Manual Excerpt", "legacy_manual", "Archives", "superseded", "1999-03-01", "2000-03-01", "public_internal", "en"),
)


def generate_synthetic_documents(force: bool = False) -> list[Path]:
    settings = get_settings()
    markdown_dir = settings.generated_corpus_dir / "source_markdown"
    generated_dir = settings.generated_corpus_dir / "generated"
    tables_dir = settings.generated_corpus_dir / "tables"
    for directory in (markdown_dir, generated_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for spec in SPECS:
        markdown_path = markdown_dir / spec.markdown_filename
        if force or not markdown_path.exists():
            markdown_path.write_text(_markdown_for(spec), encoding="utf-8")
        paths.append(markdown_path)

        generated_path = generated_dir / spec.filename
        if force or not generated_path.exists():
            if generated_path.suffix.lower() == ".docx":
                _write_docx(spec, generated_path)
            elif generated_path.suffix.lower() == ".pdf":
                _write_pdf(spec, generated_path)

    table_path = tables_dir / "doctrine_review_tracker.csv"
    if force or not table_path.exists():
        _write_tracker_csv(table_path)
    paths.append(table_path)
    return paths


def spec_by_doc_id(doc_id: str) -> SyntheticDocSpec:
    for spec in SPECS:
        if spec.doc_id == doc_id:
            return spec
    raise KeyError(doc_id)


def spec_by_filename(filename: str) -> SyntheticDocSpec:
    stem = Path(filename).stem
    return spec_by_doc_id(stem)


def source_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(spec: SyntheticDocSpec, canonical_path: str) -> dict[str, object]:
    return {
        "doc_id": spec.doc_id,
        "title": spec.title,
        "doc_family": spec.doc_family,
        "version": spec.version,
        "status": spec.status,
        "effective_date": spec.effective_date,
        "owner": spec.owner,
        "review_due": spec.review_due,
        "access_level": spec.access_level,
        "language": spec.language,
        "source_type": spec.source_type,
        "canonical_source": canonical_path,
        "allowed_roles": _allowed_roles(spec.access_level),
        "created_at": "2026-05-06T00:00:00Z",
    }


def _markdown_for(spec: SyntheticDocSpec) -> str:
    metadata = yaml.safe_dump(_metadata(spec, f"generated/{spec.filename}"), sort_keys=False).strip()
    sections: list[str] = []
    for section_id, (section, page, body) in enumerate(spec.sections, start=1):
        sections.append(f"## {section}\n\nPage: {page}\nSection ID: S{section_id}\n\n{body.strip()}\n")
    return f"---\n{metadata}\n---\n\n# {spec.title}\n\n" + "\n".join(sections)


def _allowed_roles(access_level: str) -> list[str]:
    if access_level == "restricted":
        return ["planning_lead", "admin"]
    return ["planning_analyst", "planning_lead", "auditor", "admin"]


def _write_docx(spec: SyntheticDocSpec, path: Path) -> None:
    doc = DocxDocument()
    doc.add_heading(spec.title, level=0)
    doc.add_paragraph(f"Document ID: {spec.doc_id}")
    doc.add_paragraph(f"Status: {spec.status}")
    doc.add_paragraph(f"Access level: {spec.access_level}")
    doc.add_paragraph(f"Effective date: {spec.effective_date}")
    for section, page, body in spec.sections:
        doc.add_heading(section, level=1)
        doc.add_paragraph(f"Page: {page}")
        doc.add_paragraph(body)
    doc.save(path)


def _write_pdf(spec: SyntheticDocSpec, path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(72, y, spec.title[:80])
    y -= 24
    pdf.setFont("Helvetica", 9)
    pdf.drawString(72, y, f"Document ID: {spec.doc_id} | Status: {spec.status} | Access: {spec.access_level}")
    y -= 28
    for section, page, body in spec.sections:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(72, y, f"SECTION: {section} | PAGE: {page}")
        y -= 16
        pdf.setFont("Helvetica", 9)
        for line in _wrap(body, width=92):
            if y < 72:
                pdf.showPage()
                y = height - 72
            pdf.drawString(72, y, line)
            y -= 13
        y -= 12
    pdf.save()


def _write_tracker_csv(path: Path) -> None:
    headers = (
        "doc_id",
        "title",
        "doc_family",
        "owner",
        "status",
        "effective_date",
        "next_review_due",
        "access_level",
        "language",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(TRACKER_ROWS)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        proposed = " ".join([*current, word])
        if len(proposed) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines
