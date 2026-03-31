FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    default-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-api.txt

COPY data/spark_output/ ./data/spark_output/
COPY models/ ./models/
COPY api/ ./api/
COPY drift_detection/ ./drift_detection/

EXPOSE 8000

CMD ["python3", "api/main.py"]