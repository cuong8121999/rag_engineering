import re
import emoji
import json
import great_expectations as gx
from typing import Dict, List, Any, Optional, Union
from minio import Minio
import logging
from io import BytesIO
from great_expectations.datasource.fluent.pandas_datasource import _PandasDataAsset
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# Initialize ChromaDB client with persistent storage
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Global client for reuse
_minio_client = None
_chromadb_client = None


def get_minio_client(
    endpoint_url: str = "minio:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    force_new: bool = False,
) -> Minio:
    """
    Create and return a Minio client with improved connection handling

    Args:
        endpoint_url: MinIO server endpoint URL
        access_key: MinIO access key
        secret_key: MinIO secret key
        force_new: Force creating a new client instead of reusing existing one

    Returns:
        Minio client object

    Raises:
        ConnectionError: If unable to connect to MinIO after max_retries
    """
    global _minio_client

    if not force_new and _minio_client is not None:
        # Reuse existing client if available
        return _minio_client

    logging.info(f"Connecting to MinIO server at {endpoint_url}")

    try:
        # Create the MinIO client
        minio_client = Minio(
            endpoint=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,  # Using HTTP instead of HTTPS for local development
        )

        # Test the connection
        buckets = minio_client.list_buckets()
        bucket_names = [b.name for b in buckets]
        logging.info(
            f"Successfully connected to MinIO. Available buckets: {bucket_names}"
        )

        # Cache for reuse
        _minio_client = minio_client
        return minio_client

    except Exception as e:
        return None


def _read_json_file(
    minio_client: Minio, input_file: str, bucket_name: str
) -> Optional[Union[List, Dict]]:
    """Helper function to read JSON from MinIO or local file"""

    try:
        if not minio_client:
            logging.warning("Failed to create MinIO client for reading")
        else:
            logging.info(f"Retrieving {input_file} from MinIO bucket '{bucket_name}'")

            # Check if object exists
            try:
                minio_client.stat_object(bucket_name, input_file)
            except Exception as e:
                logging.warning(
                    f"File {input_file} not found in bucket {bucket_name}. Error: {e}"
                )

            # Get object with proper handling
            response = minio_client.get_object(bucket_name, input_file)
            data = json.load(response)
            response.close()  # Explicitly close to free resources
            logging.info(
                f"Successfully retrieved data from MinIO: {bucket_name}/{input_file}"
            )
            return data

    except Exception as e:
        logging.error(f"Error retrieving file from MinIO: {e}")
        return None


def clean_text(text):
    """Applies text cleaning process to extracted text."""
    text = text.lower()  # Convert to lowercase
    # text = re.sub(
    #     r"[^a-z0-9\s-]", "", text
    # )  # Remove special characters (keep numbers and '-')
    # text = unidecode.unidecode(text)  # Normalize Unicode
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = emoji.replace_emoji(text, replace="")  # Remove emojis

    return text


def _write_output_data(
    minio_client: Minio, input_data: List[Dict], output_file: str, output_bucket: str
) -> bool:
    """Write output data to MinIO or local file"""
    try:
        if not minio_client:
            logging.warning("Failed to create MinIO client for writing")
        else:

            # Check if bucket exists
            if not minio_client.bucket_exists(output_bucket):
                logging.info(f"Creating bucket '{output_bucket}'...")
                minio_client.make_bucket(output_bucket)

            # Convert to JSON and prepare for upload
            json_data = json.dumps(input_data, ensure_ascii=False, indent=4)
            json_bytes = BytesIO(json_data.encode("utf-8"))
            data_length = len(json_data.encode("utf-8"))

            # Upload to MinIO
            minio_client.put_object(
                bucket_name=output_bucket,
                object_name=output_file,
                data=json_bytes,
                length=data_length,
                content_type="application/json",
            )
            logging.info(
                f"Cleaned data has been saved to MinIO: {output_bucket}/{output_file}"
            )
            return True

    except Exception as e:
        logging.error(f"Error saving to MinIO: {e}")
        return False


def _process_articles(data: List[Dict]) -> List[Dict]:
    """Process and clean article data with content chunking"""
    cleaned_data = []

    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Increased from 200 for more meaningful chunks
        chunk_overlap=20,  # Increased overlap for better context
    )
    logging.info("Initial Text Splitter Configuration")

    # Field mapping for extraction
    field_mappings = {
        "headline": ("title", True),
        "description": ("description", True),
        "datePublished": ("published_date", False),
        "@type": ("type", False),
    }

    # Process each article
    for article in data:
        # First extract all metadata (everything except content)
        article_metadata = {}

        logging.info("Extracting metadata from article")
        # Extract mapped fields as metadata
        for source_field, (target_field, should_clean) in field_mappings.items():
            if source_field in article:
                value = article[source_field]
                if should_clean:
                    value = clean_text(value)
                article_metadata[target_field] = value

        logging.info("Extracting URL and tags from article")
        # Extract URL (special case with nested structure)
        if "mainEntityOfPage" in article and "@id" in article["mainEntityOfPage"]:
            article_metadata["url"] = article["mainEntityOfPage"]["@id"]

        # Extract tags (special case with potential different formats)
        if "article:tag" in article:
            tags = article["article:tag"]
            if isinstance(tags, str):
                tags = tags.split(",")
            # Clean each tag
            article_metadata["tags"] = [clean_text(tag) for tag in tags]

        logging.info("Extracting article content and splitting into chunks")
        # Now handle content chunking
        if "articleContent" in article:
            content = article["articleContent"]
            if content:
                # Clean the content first
                cleaned_content = clean_text(content)

                # Split into chunks
                chunks = text_splitter.split_text(cleaned_content)

                logging.info(
                    "Adding article chunk content and metadata to cleaned data"
                )
                # Create a separate entry for each chunk with all metadata
                for i, chunk in enumerate(chunks):
                    chunk_entry = article_metadata.copy()  # Copy all metadata
                    chunk_entry["content"] = chunk  # Add this specific chunk
                    chunk_entry["chunk_id"] = i  # Add chunk identifier
                    chunk_entry["total_chunks"] = len(chunks)  # Add total chunks info

                    cleaned_data.append(chunk_entry)
            else:
                # If no content, still add the article with its metadata
                article_metadata["content"] = ""
                cleaned_data.append(article_metadata)
        else:
            # If no content field, still add the article with its metadata
            article_metadata["content"] = ""
            cleaned_data.append(article_metadata)

    return cleaned_data


context = gx.get_context(mode="file")
site_name = "rag_docs_site"

if not site_name in context.list_data_docs_sites():
    print(f"{site_name} not in the context. Try to add it")
    base_directory = context.list_data_docs_sites()["local_site"]["store_backend"][
        "base_directory"
    ]

    site_config = {
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            "base_directory": base_directory,
        },
    }
    context.add_data_docs_site(site_name=site_name, site_config=site_config)
else:
    print(f"{site_name} already in the context")

actions = [
    gx.checkpoint.actions.UpdateDataDocsAction(
        name="update_my_site", site_names=[site_name]
    )
]

print("Great Expectation Context Information")
print(context)

pandas_datasource = context.data_sources.add_or_update_pandas(name="rag_datasource")


def create_test_checkpoint(
    suite: gx.ExpectationSuite, table_name: str, data_asset: _PandasDataAsset
):
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"{table_name} batch definition"
    )

    validation_definition = gx.ValidationDefinition(
        name=f"{table_name} Validation Definition",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add_or_update(validation_definition)

    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=f"{table_name} Checkpoint",
            validation_definitions=[validation_definition],
            actions=actions,
        )
    )

    return checkpoint


# Define persistence directory
persist_directory = "./chroma_db"


def get_chromadb_client(force_new: bool = False) -> chromadb.PersistentClient:
    """
    Create and return a ChromaDB client with improved connection handling

    Args:
        None

    Returns:
        ChromaDB client object

    Raises:
        ConnectionError: If unable to connect to ChromaDB after max_retries
    """
    global _chromadb_client

    if not force_new and _chromadb_client is not None:
        # Reuse existing client if available
        return _chromadb_client

    try:
        # Create client with persistence configuration
        client = chromadb.PersistentClient(path=persist_directory)

        # Cache for reuse
        _chromadb_client = client
        return client

    except Exception as e:
        logging.error(f"Error creating ChromaDB client: {e}")
        return None
