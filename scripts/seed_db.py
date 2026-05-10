#!/usr/bin/env python3
"""
scripts/seed_db.py
════════════════════
Seeds ChromaDB with sample documents for retrieval testing.
Run once after docker compose up:
  python scripts/seed_db.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.agents.retrieval import _get_chroma_collection, _seed_collection

def main():
    print("Seeding ChromaDB knowledge base...")
    collection = _get_chroma_collection()
    count = collection.count()
    print(f"Collection '{collection.name}' already has {count} documents.")
    if count == 0:
        _seed_collection(collection)
        print(f"Seeded {collection.count()} documents.")
    else:
        print("Skipping seed — collection already populated.")
    print("Done.")

if __name__ == "__main__":
    main()
