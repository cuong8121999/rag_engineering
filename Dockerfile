FROM apache/airflow:2.10.5

USER root

RUN sudo apt update && apt install -y wget unzip curl chromium chromium-driver

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt