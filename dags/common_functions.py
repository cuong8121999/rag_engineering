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
    """Advanced text cleaning process for Vietnamese news articles"""
    if not isinstance(text, str):
        return ""

    # Step 1: Remove all video player technical information more aggressively
    # Remove all video player sections completely
    text = re.sub(
        r"Video Player is loading.*?End of dialog window\.",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove any remaining video technical data
    text = re.sub(r"XemHiện tại\d+:\d+.*?Tiến trình: \d+%", "", text, flags=re.DOTALL)
    text = re.sub(r"Đã tải:.*?Tắt tiếng", "", text, flags=re.DOTALL)
    text = re.sub(r"Tỷ lệ phát lại.*?Chương mục", "", text, flags=re.DOTALL)
    text = re.sub(
        r"descriptions off.*?Audio Track", "", text, flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(
        r"This is a modal window.*?window\.", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"TextColorWhite.*?Font Size", "", text, flags=re.DOTALL)

    # Step 2: Remove HTML-like tags more thoroughly
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 3: Normalize punctuation (keep periods and important punctuation)
    text = re.sub(r"[-–—]", " - ", text)  # Keep hyphens with spaces
    text = re.sub(r",", ", ", text)  # Add space after commas
    text = re.sub(r"\.", ". ", text)  # Add space after periods
    text = re.sub(r":", ": ", text)  # Add space after colons

    # Step 4: Restore proper Vietnamese accents handling
    # This is just basic - you might need more specialized handling depending on your needs

    # Step 5: Fix spacing around quotes
    text = re.sub(r'"([^"]*)"', r' "\1" ', text)

    # Step 6: Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Step 7: Remove any remaining boilerplate phrases
    phrases_to_remove = [
        "videoap",
        "video:",
        "ap",
        "ảnh:",
        "nguồn video:",
        "toàn màn hình",
        "tiến trình",
        "chương mục",
        "font family",
        "window color",
        "background color",
        "edge style",
        "text edge style",
    ]
    for phrase in phrases_to_remove:
        text = re.sub(rf"{phrase}.*?\s", " ", text, flags=re.IGNORECASE)

    # Step 8: Final cleaning
    text = text.lower()  # Convert to lowercase
    text = emoji.replace_emoji(text, replace="")  # Remove emoji
    text = re.sub(r"\s+", " ", text).strip()  # Final whitespace normalization

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
        chunk_size=500,
        chunk_overlap=100,
    )

    # Enhanced field mappings to capture more metadata
    field_mappings = {
        "headline": ("title", True),
        "description": ("description", True),
        "datePublished": ("published_date", False),
        "dateModified": ("last_modified", False),
        "@type": ("category", False),
        "keywords": ("keywords", True),
        "news_keywords": ("news_keywords", True),
    }

    for article in data:
        # Extract metadata
        article_metadata = {}

        # Extract mapped fields
        for source_field, (target_field, should_clean) in field_mappings.items():
            if source_field in article and article[source_field]:
                value = article[source_field]
                if should_clean and isinstance(value, str):
                    value = clean_text(value)
                article_metadata[target_field] = value

        # Extract URL
        if "mainEntityOfPage" in article and "@id" in article["mainEntityOfPage"]:
            article_metadata["url"] = article["mainEntityOfPage"]["@id"]

        # Better author handling
        if "author" in article and isinstance(article["author"], dict):
            author_data = article["author"]
            author_info = {}
            if "name" in author_data:
                author_info["name"] = author_data["name"]
            if "profile_link" in author_data and author_data["profile_link"]:
                author_info["profile"] = author_data["profile_link"]
            if author_info:
                article_metadata["author"] = author_info

        # Extract tags with better handling
        if "article:tag" in article:
            tags = article["article:tag"]
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            elif isinstance(tags, list):
                tags = [t for t in tags if isinstance(t, str)]
            article_metadata["tags"] = [clean_text(tag) for tag in tags]

        # Handle related articles
        if "relatedArticles" in article and isinstance(
            article["relatedArticles"], list
        ):
            related = []
            for rel_article in article["relatedArticles"]:
                if "title" in rel_article and "link" in rel_article:
                    related.append(
                        {
                            "title": (
                                clean_text(rel_article["title"])
                                if rel_article["title"]
                                else ""
                            ),
                            "link": rel_article["link"],
                        }
                    )
            if related:
                article_metadata["related_articles"] = related

        # Clean articleContent thoroughly
        if "articleContent" in article and article["articleContent"]:
            content = article["articleContent"]

            # Remove video player text and other boilerplate
            content = re.sub(r"Video Player is loading\.[^\n]*", "", content)
            content = re.sub(r"This is a modal window\.[^\n]*", "", content)
            content = re.sub(r"Beginning of dialog window\.[^\n]*", "", content)
            content = re.sub(r"End of dialog window\.[^\n]*", "", content)

            # Remove technical video-related content
            content = re.sub(
                r"XemHiện tại[0-9:]+/Thời lượng[0-9:].*?Tiến trình: [0-9%]+",
                "",
                content,
            )

            # Remove color settings and other technical params
            content = re.sub(r"TextColorWhite.*?End of dialog window\.", "", content)

            # Clean content
            cleaned_content = clean_text(content)

            # Split into chunks
            chunks = text_splitter.split_text(cleaned_content)

            # Create entries for chunks
            for i, chunk in enumerate(chunks):
                # Skip empty chunks
                if not chunk.strip():
                    continue

                chunk_entry = article_metadata.copy()
                chunk_entry["content"] = chunk
                chunk_entry["chunk_id"] = i
                chunk_entry["total_chunks"] = len(chunks)
                cleaned_data.append(chunk_entry)
        else:
            # Add metadata even without content
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
chroma_directory = "./chroma_db"


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
        client = chromadb.PersistentClient(path=chroma_directory)

        # Cache for reuse
        _chromadb_client = client
        return client

    except Exception as e:
        logging.error(f"Error creating ChromaDB client: {e}")
        return None
