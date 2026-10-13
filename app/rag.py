from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app import config


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


def _get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=_get_embeddings(),
        pinecone_api_key=config.PINECONE_API_KEY,
    )


def ingest_help_docs(force: bool = False) -> int:
    """Load every .md file from data/help_docs/ and upsert to Pinecone.

    Args:
        force: if False, skips ingest when the index already has vectors.

    Returns:
        Number of documents upserted.
    """
    vectorstore = _get_vectorstore()

    if not force:
        existing = vectorstore.similarity_search("test", k=1)
        if existing:
            print("Index already populated. Use force=True to re-ingest.")
            return 0

    docs = []
    for md_file in sorted(config.HELP_DOCS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": md_file.name,
                    "title": md_file.stem.replace("_", " ").title(),
                },
            )
        )

    vectorstore.add_documents(docs)
    print(f"Upserted {len(docs)} documents to '{config.PINECONE_INDEX_NAME}'.")
    return len(docs)


def retrieve(query: str, k: int = 4) -> list[Document]:
    """Retrieve the top-K most relevant help docs for a query."""
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search(query, k=k)
