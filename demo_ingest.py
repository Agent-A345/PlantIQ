import shutil
from pathlib import Path
from ingest import IngestConfig, ingest_directory, query_collection

def main():
    src_dir = Path("./demo_docs_refinery")
    db_dir  = Path("./voyage_faiss_db")

    if not src_dir.exists():
        print(f"ERROR: {src_dir} not found.")
        return

    if db_dir.exists():
        shutil.rmtree(db_dir)

    print("PlantIQ — Ingesting Vadodara Refinery Documents")
    print("=" * 50)

    cfg = IngestConfig(
        input_dir=str(src_dir),
        db_dir=str(db_dir),
        chunk_size=300,
        chunk_overlap=40,
        embedding_backend="voyage",
    )

    summary = ingest_directory(cfg)
    print(f"\nSummary: {summary}")
    print(f"\nDone. Run: python server.py")

if __name__ == "__main__":
    main()
