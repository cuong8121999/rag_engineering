# Prequesites:
- Create chrome_db, config, logs, and minio_data folders
- Docker engine available

# Build custom Airflow image
Run command:
docker build -t custom-airflow:latest .

# Build docker services
Run command:
docker compose up -d

# Check running containers
Run command:
docker ps

# Access Airflow and MinIO UI
The URLs:
- Airflow - http://localhost:8080/login
     - username: airflow
     - pwd: airflow
- MinIO   - http://localhost:9001/login
     - username: minioadmin
     - pwd: minioadmin

# Create MinIO buckets
Create landing and staging buckets via MinIO UI