#!/usr/bin/env python3
"""Retrieval harness for the veterinary RAG knowledge base.

Given a clinical query, return the top-k grounding chunks with similarity
scores and their citations. This is the thin retrieval layer that the LangChain
pipeline (research proposal Phase 2) will wrap to ground MedGemma's treatment
recommendations and to compute the hallucination-rate metric (spec section 6).

The metadata schema written by build_chroma_index.py supports constraining the
retrieval by disease, species, and section -- essential because a treatment
recommendation for a dog must not be grounded in cattle-only literature.

Examples
--------
    python scripts/query_rag.py "buparvaquone dose for east coast fever"
    python scripts/query_rag.py "anaemia and weight loss in cattle" --species cattle -k 3
    python scripts/query_rag.py "vaccine schedule" --disease "Canine Parvovirus"
    python scripts/query_rag.py "treatment" --section treatment --species goat
    python scripts/query_rag.py "fever and diarrhoea" --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "data" / "rag_knowledge_base"
CHROMA_DIR = KB_DIR / "chroma_db"
COLLECTION_NAME = "vet_kb"


def build_where(disease: str | None, species: str | None,
                section: str | None) -> dict | None:
    """Compose a Chroma `where` filter from optional constraints."""
    clauses: list[dict] = []
    if disease:
        clauses.append({"disease": disease})
    if section:
        clauses.append({"section": section})
    if species:
        # per-species boolean flag written by build_chroma_index.flatten_metadata
        clauses.append({f"species_{species}": True})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query(text: str, k: int, where: dict | None) -> list[dict]:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        print(
            "ERROR: chromadb is not installed. Install with:\n"
            "    pip install chromadb sentence-transformers",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if not CHROMA_DIR.exists():
        print(
            f"ERROR: no ChromaDB store at {CHROMA_DIR}.\n"
            "Build it first: python scripts/build_chroma_index.py",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Import here so the embedding model name stays a single source of truth.
    from build_chroma_index import EMBEDDING_MODEL

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)

    res = collection.query(
        query_texts=[text],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "score": round(1.0 - float(dist), 4),  # cosine distance -> similarity
            "distance": round(float(dist), 4),
            "doc_id": meta.get("doc_id"),
            "disease": meta.get("disease"),
            "species": meta.get("species"),
            "section": meta.get("section"),
            "title": meta.get("title"),
            "citation": meta.get("citation"),
            "url": meta.get("url"),
            "text": doc,
        })
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Clinical query text.")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of chunks (default 5).")
    parser.add_argument("--disease", help="Constrain to an exact disease name.")
    parser.add_argument("--species", help="Constrain to a species (cattle, goat, sheep, dog, cat).")
    parser.add_argument("--section", help="Constrain to a section (aetiology, clinical_signs, "
                                          "diagnosis, treatment, prevention, epidemiology_east_africa).")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON results.")
    args = parser.parse_args()

    where = build_where(args.disease, args.species, args.section)
    hits = query(args.query, args.top_k, where)

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return

    if not hits:
        print("No results (check your filters).")
        return

    print(f"Query: {args.query!r}")
    if where:
        print(f"Filter: {where}")
    print(f"Top {len(hits)} of {args.top_k} requested:\n")
    for rank, h in enumerate(hits, start=1):
        print(f"[{rank}] score={h['score']:.4f}  {h['doc_id']}  "
              f"({h['disease']} / {h['section']} / {h['species']})")
        print(f"    {h['title']}")
        text = h["text"]
        snippet = text if len(text) <= 320 else text[:317] + "..."
        print(f"    {snippet}")
        print(f"    SOURCE: {h['citation']}")
        print(f"    URL   : {h['url']}\n")


if __name__ == "__main__":
    main()
