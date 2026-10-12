from pinecone import Pinecone, ServerlessSpec

from app import config


def ensure_index_exist() -> None:
    """Creates a Pinecine Serverless index if it does not already exist.

    safe to run repeatedly - checks first and then creates if needed.
    """

    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # List existing indexes (returns IndexList; iterate over .names())
    existing = pc.list_indexes().names()

    if config.PINECONE_INDEX_NAME in existing:
        print(f"Index '{config.PINECONE_INDEX_NAME}' already exists. Nothing to do.")
        return

    print(
        f"Creating index '{config.PINECONE_INDEX_NAME}' "
        f"({config.EMBEDDING_DIMENSION} dims, cosine, "
        f"{config.PINECONE_CLOUD}/{config.PINECONE_REGION})..."
    )

    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        ),
    )

    print(f"Index {config.PINECONE_INDEX_NAME} has been successfully created.")


if __name__ == "__main__":
    ensure_index_exist()
