"""
Turns knowledge-base/*.md into a flat list of "chunks": one chunk per
section (## heading) of one document, tagged with that document's front
matter metadata.
"""

from __future__ import annotations
import os
import re
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    doc_id: str
    filename: str
    title: str
    heading: str
    text: str
    status: str
    policy_authority: str
    audience: str
    effective_date: Optional[str] = None
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None
    customer_answering: bool = True

    @property
    def is_authoritative(self) -> bool:
        """Can this chunk be used as the basis for a customer-facing claim?"""
        return (
            self.status == "active"
            and self.policy_authority == "official"
            and self.customer_answering is not False
        )

    def citation(self) -> str:
        return f"{self.filename}" + (f" — {self.heading}" if self.heading else "")


_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    m = _FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)
    return meta, body


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split on '## ' headings into (heading, text) pairs."""
    sections = []
    matches = list(_HEADING_RE.finditer(body))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))
    return sections


def load_chunks(kb_dir: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for filename in sorted(os.listdir(kb_dir)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(kb_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = _parse_front_matter(raw)
        for heading, text in _split_sections(body):
            chunks.append(
                Chunk(
                    doc_id=meta.get("document_id", filename),
                    filename=filename,
                    title=meta.get("title", filename),
                    heading=heading,
                    text=text,
                    status=meta.get("status", "active"),
                    policy_authority=meta.get("policy_authority", "official"),
                    audience=meta.get("audience", "customer"),
                    effective_date=meta.get("effective_date"),
                    supersedes=meta.get("supersedes"),
                    superseded_by=meta.get("superseded_by"),
                    customer_answering=meta.get("customer_answering", True),
                )
            )
    return chunks


if __name__ == "__main__":
    cs = load_chunks(os.path.join(os.path.dirname(__file__), "..", "knowledge-base"))
    print(f"Loaded {len(cs)} chunks from {len(set(c.filename for c in cs))} documents")
    for c in cs[:3]:
        print("-", c.filename, "|", c.heading, "| authoritative:", c.is_authoritative)