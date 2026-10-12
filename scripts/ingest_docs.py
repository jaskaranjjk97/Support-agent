from app.rag import ingest_help_docs

if __name__ == "__main__":
    n = ingest_help_docs(force=False)
    print(f"Ingest complete: {n} docs.")
