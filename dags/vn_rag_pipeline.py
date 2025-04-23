from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import existing libraries
from sentence_transformers import SentenceTransformer
import pandas as pd
from common_functions import (
    get_minio_client,
    get_chromadb_client,
    _read_json_file,
    _write_output_data,
    _process_articles,
    pandas_datasource,
    create_test_checkpoint,
    Minio,
    context,
)
from ingest_data import *

from great_expectations import RunIdentifier
import great_expectations.expectations as gxe
import great_expectations as gx
import chromadb
import logging


# Keep the SentenceTransformer wrapper
class SentenceTransformerEmbeddings:
    """Wrapper class for SentenceTransformer that follows ChromaDB's EmbeddingFunction interface."""

    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input):
        # The expected signature by ChromaDB: (self, input)
        return self.model.encode(input).tolist()


# Define persistence directory
persist_directory = "./chroma_db"

# Define default DAG arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# Create the DAG
with DAG(
    "vn_rag_pipeline",
    default_args=default_args,
    description="Vietnamese RAG Pipeline for News Articles",
    schedule_interval=None,  # Set to None for manual triggering
    start_date=datetime(2025, 4, 14),
    catchup=False,
    tags=["rag", "vietnamese", "embedding"],
) as dag:

    def ingest_news_task(**context):
        """Task to ingest news on vnexpress.vn"""
        output_bucket = context["params"].get("output_bucket", "landing")
        # Initialize MinIO client
        minio_client = get_minio_client()
        if not minio_client:
            raise ValueError("Failed to initialize MinIO client")

        # Save all articles' metadata to a single JSON file with a timestamped filename
        current_timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )  # Format: YYYYMMDD_HHMMSS
        output_file = f"{current_timestamp}_articles_metadata.json"
        try:
            ingest_vnexpress(minio_client, output_file=output_file)

            # Pass the output filename and bucket to the next task via XCom
            context["ti"].xcom_push(key="ingested_file", value=output_file)
            context["ti"].xcom_push(key="ingested_bucket", value=output_bucket)

            return f"Successfully ingested articles to {output_bucket}/{output_file}"
        except Exception as e:
            raise Exception(f"Error ingesting articles: {e}")

    # Task 1: Clean JSON Data
    def clean_json_data_task(**context):
        """Task to clean and process JSON data"""
        # Get parameters from previous task via XCom
        ti = context["ti"]
        input_file = ti.xcom_pull(task_ids="ingest_news", key="ingested_file")
        input_bucket = ti.xcom_pull(task_ids="ingest_news", key="ingested_bucket")

        # Fall back to params if XCom values aren't available
        if not input_file:
            raise ValueError("Input file not found")
        if not input_bucket:
            input_bucket = context["params"].get("input_bucket", "landing")

        output_file = context["params"].get("output_file", f"cleaned_{input_file}")
        output_bucket = context["params"].get("output_bucket", "staging")

        minio_client = get_minio_client()
        # Get MinIO client
        if not minio_client:
            raise ValueError("Failed to initialize MinIO client")

        # Call the original function
        cleaned_data = clean_json_data(
            minio_client,
            input_file=input_file,
            output_file=output_file,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
        )

        if cleaned_data is None:
            raise ValueError(f"Failed to clean data from {input_file}")

        # Pass the output filename to the next task via XCom
        context["ti"].xcom_push(key="cleaned_file", value=output_file)
        context["ti"].xcom_push(key="cleaned_bucket", value=output_bucket)
        return f"Successfully cleaned {len(cleaned_data)} articles"

    # Task 2: Validate JSON Data
    def validate_json_data_task(**context):
        """Task to validate the cleaned JSON data"""
        # Get the output filename from the previous task via XCom
        ti = context["ti"]
        input_file = ti.xcom_pull(task_ids="clean_json_data", key="cleaned_file")
        input_bucket = ti.xcom_pull(task_ids="clean_json_data", key="cleaned_bucket")
        table_name = "cleaned_articles"

        minio_client = get_minio_client()
        # Get MinIO client
        if not minio_client:
            raise ValueError("Failed to initialize MinIO client")

        # Call the original function
        result = validate_json_data(
            minio_client,
            input_file=input_file,
            input_bucket=input_bucket,
            table_name=table_name,
        )

        if result is None:
            raise ValueError(f"Validation failed for {input_file}")

        return "Data validation completed successfully"

    # Task 3: Embed Text and Save to Vector DB
    def embed_text_and_save_vectordb_task(**context):
        """Task to create embeddings and save to vector database"""
        # Get the validated filename from the previous task via XCom
        ti = context["ti"]
        input_file = ti.xcom_pull(task_ids="clean_json_data", key="cleaned_file")
        input_bucket = ti.xcom_pull(task_ids="clean_json_data", key="cleaned_bucket")
        model_name = context["params"].get(
            "model_name", "dangvantuan/vietnamese-document-embedding"
        )
        embedded_field = "content"

        minio_client = get_minio_client()
        chromadb_client = get_chromadb_client()
        # Get MinIO and ChromaDB clients
        if not minio_client or not chromadb_client:
            raise ValueError("Failed to initialize clients")

        # Call the original function
        result = embed_text_and_save_vectordb(
            minio_client,
            chromadb_client,
            input_file,
            input_bucket,
            model_name,
            embedded_field,
        )

        if result.get("status") == "error":
            raise ValueError(f"Embedding creation failed: {result.get('reason')}")

        return f"Created embeddings for {result.get('documents_processed')} documents"

    # Keep the original functions
    def clean_json_data(
        minio_client, input_file, output_file, input_bucket, output_bucket
    ):
        """Original clean_json_data function implementation"""
        # Implementation stays the same
        data = None
        try:
            data = _read_json_file(minio_client, input_file, input_bucket)
            if data is None:
                return None
            cleaned_data = _process_articles(data)
            success = _write_output_data(
                minio_client, cleaned_data, output_file, output_bucket
            )
            if success:
                logging.info(f"Processed {len(cleaned_data)} articles")
                return cleaned_data
            return None
        except Exception as e:
            logging.error(f"An error occurred while cleaning the data: {e}")
            return None

    def validate_json_data(minio_client, input_file, input_bucket, table_name):
        """Original validate_json_data function implementation"""
        # Implementation stays the same
        try:
            data = _read_json_file(minio_client, input_file, input_bucket)
            if data is None:
                return None
            df = pd.DataFrame(data)

            suite = context.suites.add_or_update(
                gx.ExpectationSuite(f"{table_name} Suite")
            )

            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="content"))

            suite.save()

            data_asset = pandas_datasource.add_dataframe_asset(name=table_name)

            checkpoint = create_test_checkpoint(
                suite=suite, table_name=table_name, data_asset=data_asset
            )

            checkpoint_result = checkpoint.run(
                batch_parameters={"dataframe": df},
                run_id=RunIdentifier(run_name="rag_validation"),
            )
            return {"status": "success", "validation_result": checkpoint_result.success}
        except Exception as e:
            logging.error(f"Error during validation: {e}")
            return None

    def embed_text_and_save_vectordb(
        minio_client,
        chromadb_client,
        input_file,
        input_bucket,
        model_name,
        embedded_field,
    ):
        """Original embed_text_and_save_vectordb function implementation"""
        # Implementation stays the same but with the empty document filtering fix
        try:
            data = _read_json_file(minio_client, input_file, input_bucket)
            if data is None:
                return {"status": "error", "reason": "Failed to read input file"}

            embedding_function = SentenceTransformerEmbeddings(model_name)
            collection_name = f"chunk_articles_{input_file.split('_')[0]}"
            collection = chromadb_client.get_or_create_collection(
                collection_name, embedding_function=embedding_function
            )

            valid_docs = []
            valid_metadatas = []

            for article in data:
                if (
                    embedded_field in article
                    and article[embedded_field]
                    and isinstance(article[embedded_field], str)
                ):
                    valid_docs.append(article[embedded_field])
                    metadata = {
                        k: v
                        for k, v in article.items()
                        if k != embedded_field and k != "tags" and v is not None
                    }
                    for key, value in metadata.items():
                        if not isinstance(value, (str, int, float, bool)):
                            metadata[key] = str(value)
                    valid_metadatas.append(metadata)

            if valid_docs:
                ids = [f"doc_{i}" for i in range(len(valid_docs))]
                batch_size = 100

                for i in range(0, len(valid_docs), batch_size):
                    end_idx = min(i + batch_size, len(valid_docs))

                    # Get batch and filter empty documents
                    batch_docs = valid_docs[i:end_idx]
                    batch_metadatas = valid_metadatas[i:end_idx]
                    batch_ids = ids[i:end_idx]

                    valid_indices = [
                        j for j, doc in enumerate(batch_docs) if doc and doc.strip()
                    ]
                    if not valid_indices:
                        continue

                    filtered_docs = [batch_docs[j] for j in valid_indices]
                    filtered_metadatas = [batch_metadatas[j] for j in valid_indices]
                    filtered_ids = [batch_ids[j] for j in valid_indices]

                    collection.add(
                        documents=filtered_docs,
                        metadatas=filtered_metadatas,
                        ids=filtered_ids,
                    )

                return {
                    "status": "success",
                    "documents_processed": len(valid_docs),
                    "collection_name": collection_name,
                }
            else:
                return {
                    "status": "warning",
                    "reason": "No valid documents found",
                    "documents_processed": 0,
                }

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"status": "error", "reason": str(e)}

    # Create the actual tasks
    ingest_news = PythonOperator(
        task_id="ingest_news",
        python_callable=ingest_news_task,
        provide_context=True,
        params={
            "output_bucket": "landing",
        },
    )

    clean_task = PythonOperator(
        task_id="clean_json_data",
        python_callable=clean_json_data_task,
        provide_context=True,
    )

    validate_task = PythonOperator(
        task_id="validate_json_data",
        python_callable=validate_json_data_task,
        provide_context=True,
    )

    embed_task = PythonOperator(
        task_id="embed_text_and_save_vectordb",
        python_callable=embed_text_and_save_vectordb_task,
        provide_context=True,
        params={
            "model_name": "dangvantuan/vietnamese-document-embedding",
        },
    )

    # Define task dependencies (sequential execution)
    ingest_news >> clean_task >> validate_task >> embed_task
