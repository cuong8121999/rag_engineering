from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from airflow.models import Variable

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


# Define default DAG arguments
default_args = {
    "owner": "airflow",
    "retries": 5,
    "retry_delay": timedelta(minutes=1),
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

        # Store validation result in XCom for branching
        validation_passed = result.get("validation_result", False)
        context["ti"].xcom_push(key="validation_passed", value=validation_passed)

        if not validation_passed:
            logging.warning(
                "Data validation detected quality issues - will skip embedding"
            )

        return "Data validation completed successfully"

    def decide_if_embedding_needed(**context):
        """Decide whether to proceed with embedding based on validation results"""
        ti = context["ti"]
        validation_passed = ti.xcom_pull(
            task_ids="validate_json_data", key="validation_passed"
        )

        if validation_passed:
            logging.info("Validation passed - proceeding with embedding")
            return "embed_text_and_save_vectordb"
        else:
            logging.warning("Validation failed - skipping embedding")
            return "skip_embedding"

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

            # Check date format with regex (more precise)
            suite.add_expectation(
                gxe.ExpectColumnValuesToMatchRegex(
                    column="published_date",
                    regex=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",
                )
            )

            # Validate main article URL format
            suite.add_expectation(
                gxe.ExpectColumnValuesToMatchRegex(
                    column="url",
                    regex=r"^https://vnexpress\.net/[a-zA-Z0-9\-]+\.html$",
                )
            )

            # Ensure chunk_id is less than total_chunks
            suite.add_expectation(
                gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                    column_A="total_chunks", column_B="chunk_id", or_equal=True
                )
            )

            # Ensure chunk_id is non-negative
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(column="chunk_id", min_value=0)
            )

            # Ensure total_chunks is positive
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(column="total_chunks", min_value=1)
            )

            # Ensure content has minimum length
            suite.add_expectation(
                gxe.ExpectColumnValueLengthsToBeBetween(
                    column="content",
                    min_value=50,  # Adjust minimum content length as needed
                    mostly=0.95,  # Allow some flexibility
                )
            )

            # Content shouldn't contain video player leftover text
            suite.add_expectation(
                gxe.ExpectColumnValuesToNotMatchRegexList(
                    column="content",
                    regex_list=[
                        r"video player",
                        r"this is a modal window",
                        r"xemhiện tại",
                        r"tiến trình: [0-9]+%",
                    ],
                    mostly=0.98,  # Allow some flexibility
                )
            )

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
        collection_name="articles",  # Use consistent collection name
    ):
        """Original embed_text_and_save_vectordb function implementation"""
        # Implementation stays the same but with the empty document filtering fix
        try:
            data = _read_json_file(minio_client, input_file, input_bucket)
            if data is None:
                return {"status": "error", "reason": "Failed to read input file"}

            embedding_function = SentenceTransformerEmbeddings(model_name)

            # Use consistent collection name
            logging.info(f"Using collection: {collection_name}")
            collection = chromadb_client.get_or_create_collection(
                collection_name, embedding_function=embedding_function
            )

            # Log collection stats before adding
            initial_count = collection.count()
            logging.info(f"Collection contains {initial_count} documents before adding")

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
                timestamp = datetime.now().strftime("%Y%m%d%H%M")
                ids = [f"{timestamp}_doc_{i}" for i in range(len(valid_docs))]
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

                # Log final stats
                final_count = collection.count()
                logging.info(
                    f"Collection now contains {final_count} documents (+{final_count - initial_count})"
                )

                # Return more detailed information
                return {
                    "status": "success",
                    "documents_processed": len(valid_docs),
                    "collection_name": collection_name,
                    "initial_count": initial_count,
                    "final_count": final_count,
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

    # Create the branching task
    branch_task = BranchPythonOperator(
        task_id="check_validation_result",
        python_callable=decide_if_embedding_needed,
        provide_context=True,
        dag=dag,
    )

    # Create a task that runs when validation fails
    skip_embedding = DummyOperator(
        task_id="skip_embedding",
        dag=dag,
    )

    embed_task = PythonOperator(
        task_id="embed_text_and_save_vectordb",
        python_callable=embed_text_and_save_vectordb_task,
        provide_context=True,
        params={
            "model_name": "dangvantuan/vietnamese-document-embedding",
        },
    )

    from datahub.emitter.rest_emitter import DatahubRestEmitter
    from datahub.emitter.mce_builder import make_dataset_urn
    from datahub.emitter.mcp import MetadataChangeProposalWrapper
    from datahub.metadata.schema_classes import DatasetPropertiesClass, ChangeTypeClass
    import slugify  # python-slugify>=8.0.1

    def push_news_datasets_to_datahub_task(**context):
        ti = context["ti"]
        json_file = ti.xcom_pull(task_ids="ingest_news", key="ingested_file")
        json_bucket = ti.xcom_pull(task_ids="ingest_news", key="ingested_bucket")
        articles = _read_json_file(get_minio_client(), json_file, json_bucket) or []

        emitter = DatahubRestEmitter(
            gms_server=Variable.get("DATAHUB_GMS", default_var="http://datahub-gms:8080")
        )

        for art in articles:
            slug = slugify.slugify(art["headline"])[:200]
            ds_urn = make_dataset_urn("news", slug, "PROD")

            mcp = MetadataChangeProposalWrapper(
                entityUrn=ds_urn,
                entityType="dataset",
                aspectName="datasetProperties",
                aspect=DatasetPropertiesClass(
                    name=art["headline"],
                    description=art.get("description", ""),
                    customProperties={
                        "type":      art.get("@type", ""),
                        "desc":      art.get("description", ""),
                        "published": art.get("datePublished", ""),
                        "author":    art.get("author", {}).get("name", ""),
                        "keywords":  art.get("keywords", ""),
                    },
                ),
                changeType=ChangeTypeClass.UPSERT,
            )
            emitter.emit(mcp)

        return f"Pushed {len(articles)} datasets (minimal) to DataHub"

    push_to_datahub = PythonOperator(
        task_id="push_news_datasets_to_datahub",
        python_callable=push_news_datasets_to_datahub_task,
        provide_context=True,
    )

    ingest_news >> push_to_datahub >> clean_task >> validate_task >> branch_task
    branch_task >> [embed_task, skip_embedding]
