#!/usr/bin/env python3
"""Measure a lexical SQLite FTS5 baseline on the fixed, source-disjoint query set.

This is a separate retriever from Chroma/SentenceTransformer; its measured scores
must not be presented as Chroma performance or a RAG generation improvement.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from evaluate_retrieval import evaluate, load

ROOT = Path(__file__).resolve().parent.parent
STOP = {"a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is", "are", "be", "what", "which", "how", "can", "when", "from", "with", "does", "do", "at", "by", "into", "source", "described", "veterinary"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/secondary/text/retrieval_heldout.jsonl")
    parser.add_argument("--development", type=Path, default=ROOT / "data/secondary/text/retrieval_eval.jsonl")
    parser.add_argument("--kb", type=Path, default=ROOT / "data/rag_knowledge_base/kb_chunks.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/secondary/reports/retrieval_heldout_lexical.json")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    rows, development, chunks = load(args.input), load(args.development), load(args.kb)
    families = {row["group_id"] for row in development}
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate held-out item IDs")
    if families & {row["group_id"] for row in rows}:
        raise ValueError("Held-out source groups overlap the development set")
    known = {doc["doc_id"]: doc for doc in chunks}
    source_pages = set()
    for row in rows:
        for doc_id in row["relevant_doc_ids"]:
            if doc_id not in known:
                raise ValueError(f"Unknown reference passage: {doc_id}")
            doc = known[doc_id]
            source_pages.add(doc["source"]["url"])
            if (doc["disease"] != row.get("disease") or
                    row.get("species") not in doc["species"] or
                    (row.get("section") and doc["section"] != row["section"])):
                raise ValueError(f"Reference passage excluded by its own query filter: {doc_id}")

    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, disease UNINDEXED, species UNINDEXED, section UNINDEXED, title, text, tokenize='porter unicode61')")
        db.executemany(
            "INSERT INTO passages (doc_id,disease,species,section,title,text) VALUES (?,?,?,?,?,?)",
            [(d["doc_id"], d["disease"], "," + ",".join(d["species"]) + ",", d["section"], d["title"], d["text"]) for d in chunks],
        )

        def lexical_query(text: str, k: int, where: dict | None) -> list[dict]:
            filters = {}
            if where:
                for clause in where.get("$and", [where]):
                    filters.update(clause)
            terms = [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2 and t not in STOP]
            if not terms:
                return []
            match = " OR ".join('"' + t + '"' for t in dict.fromkeys(terms))
            sql = "SELECT doc_id FROM passages WHERE passages MATCH ?"
            params: list = [match]
            for key in ("disease", "section"):
                if key in filters:
                    sql += f" AND {key} = ?"
                    params.append(filters[key])
            species = next((key[8:] for key in filters if key.startswith("species_")), None)
            if species:
                sql += " AND species LIKE ?"
                params.append(f"%,{species},%")
            sql += " ORDER BY bm25(passages, 0.0, 0.0, 0.0, 2.0, 1.0), doc_id LIMIT ?"
            params.append(k)
            return [{"doc_id": doc_id} for (doc_id,) in db.execute(sql, params)]

        result = evaluate(rows, args.k, lexical_query)
        # Exploratory difficulty check: repeat without disease gating, retaining
        # the species and any predeclared section constraint. Keep this separate
        # from the original, predeclared filtered score.
        def without_disease(text: str, k: int, where: dict | None) -> list[dict]:
            if not where:
                return lexical_query(text, k, where)
            clauses = [clause for clause in where.get("$and", [where]) if "disease" not in clause]
            reduced = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
            return lexical_query(text, k, reduced)

        ablation = evaluate(rows, args.k, without_disease)
        result.update({
            "retriever": "SQLite FTS5 Porter BM25 (title weight 2; body weight 1)",
            "index_documents": len(chunks), "development_groups_excluded": sorted(families),
            "heldout_groups": sorted({row["group_id"] for row in rows}),
            "heldout_reference_source_pages": len(source_pages),
            "exploratory_without_disease_filter": ablation,
            "limitations": "Ten disease groups span nine unique source pages (canine and bovine babesiosis share a source); in-corpus source-derived relevance, no independent clinical judgements; scores are not Chroma scores.",
        })
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
