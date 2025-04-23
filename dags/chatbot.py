import logging
from sentence_transformers import SentenceTransformer
import chromadb


class SentenceTransformerEmbeddings:
    """Wrapper class for SentenceTransformer that follows ChromaDB's EmbeddingFunction interface."""

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input):
        # The expected signature by ChromaDB: (self, input)
        return self.model.encode(
            input
        ).tolist()  # FIXED: Use encode() and convert to list


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    persist_directory = "./chroma_db"

    try:
        # Create client with persistence configuration
        client = chromadb.PersistentClient(path=persist_directory)
    except Exception as e:
        logging.error(f"Error creating ChromaDB client: {e}")
        exit(1)

    try:
        # Use our wrapper instead of direct SentenceTransformer
        embedding_function = SentenceTransformerEmbeddings(
            "dangvantuan/vietnamese-document-embedding"
        )
        logging.info(
            "Model dangvantuan/vietnamese-document-embedding loaded successfully"
        )
    except Exception as e:
        logging.error(
            f'Failed to load embedding model "dangvantuan/vietnamese-document-embedding": {e}'
        )
        exit(1)

    collection = client.get_or_create_collection(
        "articles", embedding_function=embedding_function
    )

    result = collection.query(query_texts=["Giá vàng 22/4"], n_results=3)

    question = "Giá vàng 22/4"
    # FIXED: Process the result dictionary correctly
    print(f"\nSearch results for: '{question}'")
    print("-" * 50)

    # ChromaDB results come as a dictionary with lists
    if "documents" in result and result["documents"]:
        for i, doc in enumerate(result["documents"][0]):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc}")  # Show first 200 chars
            print("-" * 50)
    else:
        print("No results found")
