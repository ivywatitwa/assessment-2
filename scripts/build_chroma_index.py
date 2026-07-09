#!/usr/bin/env python3
"""Ingest the veterinary RAG knowledge base into a persistent ChromaDB collection.

The corpus (data/rag_knowledge_base/kb_chunks.jsonl) is the grounding substrate
for the MedGemma treatment-recommendation task (see docs/PROJECT_SPEC.md, task
T2). Each chunk is embedded with a sentence-transformers model and stored with
metadata that lets the RAG retriever constrain results by disease, species, and
section.

Usage
-----
Validate the corpus without installing chromadb (CI / low-dependency check):
    python scripts/build_chroma_index.py --dry-run

Build / rebuild the persistent index:
    python scripts/build_chroma_index.py

Query smoke-test after building:
    python scripts/query_rag.py "buparvaquone dose for east coast fever"

Design notes
------------
* EMBEDDING_MODEL is a documented constant. all-MiniLM-L6-v2 is a strong,
  lightweight default (384-dim, ~256-token context). Chunks in kb_chunks.jsonl
  are intentionally short (~80-140 words) so each fits inside that context
  window without truncation; see the knowledge-base README for the rationale.
* Chroma `where` filters match scalar metadata exactly. Because a chunk can
  apply to several species, we store species two ways: a human-readable
  comma-joined `species` string, and one boolean flag per species
  (`species_cattle`, `species_dog`, ...). Filter on the boolean flags to get
  reliable "contains this species" behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Documented configuration constants
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "all-MiniLM-L6-v2"          # sentence-transformers model id
COLLECTION_NAME = "vet_kb"
REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "data" / "rag_knowledge_base"
CHUNKS_PATH = KB_DIR / "kb_chunks.jsonl"
CHROMA_DIR = KB_DIR / "chroma_db"

REQUIRED_FIELDS = {
    "doc_id", "title", "disease", "species", "section", "text", "source",
    "provenance",
}
REQUIRED_SOURCE_FIELDS = {"citation", "type", "url"}
VALID_SECTIONS = {
    "aetiology", "clinical_signs", "diagnosis", "treatment", "prevention",
    "epidemiology_east_africa",
}


# --------------------------------------------------------------------------- #
# Loading + validation (no heavy dependencies -> works under --dry-run)
# --------------------------------------------------------------------------- #
def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}. Run scripts/build_kb_chunks.py first."
        )
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return chunks


def validate_chunks(chunks: list[dict]) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    if not chunks:
        return ["Corpus is empty."]
    for i, c in enumerate(chunks):
        tag = c.get("doc_id", f"<index {i}>")
        missing = REQUIRED_FIELDS - set(c)
        if missing:
            errors.append(f"{tag}: missing fields {sorted(missing)}")
            continue
        if c["doc_id"] in seen_ids:
            errors.append(f"{tag}: duplicate doc_id")
        seen_ids.add(c["doc_id"])
        if not isinstance(c["species"], list) or not c["species"]:
            errors.append(f"{tag}: species must be a non-empty list")
        if c["section"] not in VALID_SECTIONS:
            errors.append(f"{tag}: unknown section '{c['section']}'")
        if not isinstance(c["text"], str) or len(c["text"].split()) < 20:
            errors.append(f"{tag}: text missing or suspiciously short")
        src = c.get("source", {})
        missing_src = REQUIRED_SOURCE_FIELDS - set(src)
        if missing_src:
            errors.append(f"{tag}: source missing {sorted(missing_src)}")
        elif not str(src.get("url", "")).startswith("http"):
            errors.append(f"{tag}: source.url is not a URL")
    return errors


def flatten_metadata(chunk: dict) -> dict:
    """Chroma metadata values must be scalars. Flatten species list into a
    comma-joined string plus per-species boolean flags for reliable filtering."""
    meta = {
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "disease": chunk["disease"],
        "section": chunk["section"],
        "species": ", ".join(chunk["species"]),
        "citation": chunk["source"]["citation"],
        "source_type": chunk["source"]["type"],
        "url": chunk["source"]["url"],
        "provenance": chunk["provenance"],
    }
    for sp in chunk["species"]:
        meta[f"species_{sp}"] = True
    return meta


# --------------------------------------------------------------------------- #
# Index build (imports chromadb + sentence-transformers lazily)
# --------------------------------------------------------------------------- #
def build_index(chunks: list[dict], reset: bool = True) -> None:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:  # pragma: no cover - depends on environment
        print(
            "ERROR: chromadb is not installed. Install with:\n"
            "    pip install chromadb sentence-transformers\n"
            "Or run with --dry-run to validate the corpus without ingesting.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # collection did not exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    ids = [c["doc_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [flatten_metadata(c) for c in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingested {len(ids)} chunks into collection '{COLLECTION_NAME}'")
    print(f"Persistent store: {CHROMA_DIR}")
    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Collection count: {collection.count()}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the chunk corpus without importing chromadb / embedding it.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Add to the existing collection instead of rebuilding it.",
    )
    args = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    errors = validate_chunks(chunks)

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} error(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(2)

    diseases = sorted({c["disease"] for c in chunks})
    species = sorted({sp for c in chunks for sp in c["species"]})
    sections = sorted({c["section"] for c in chunks})
    print(f"Validated {len(chunks)} chunks: OK")
    print(f"  diseases ({len(diseases)}): {', '.join(diseases)}")
    print(f"  species  ({len(species)}): {', '.join(species)}")
    print(f"  sections ({len(sections)}): {', '.join(sections)}")

    if args.dry_run:
        print("\n--dry-run: corpus is valid; skipping ChromaDB ingestion.")
        return

    build_index(chunks, reset=not args.no_reset)


if __name__ == "__main__":
    main()
