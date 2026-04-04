# # BUILDING: docker build -t <container_name> .
# # RUNNING: docker run <container_name> <python_program_path> <config_file_path>
# # e.g. docker run --network="host" anomaly_detection
# FROM ubuntu:20.04
# RUN apt-get update -y && \
#     apt-get install -y python3-pip python3-dev
# COPY ./requirements.txt /requirements.txt
# WORKDIR /
# RUN pip3 install -r requirements.txt
# COPY . /
# # CMD ["python3", "main.py", "-c", "deployment/NAIADES_Braila.json"]
# # CMD ["python3", "main.py", "-c", "deployment/NAIADES_Braila_flow.json"]
# CMD ["python3", "main.py", "-c", "FileConfig.json"]

FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries 10 -r requirements.txt

# Copy application code
COPY ./api ./api
COPY ./src ./src
COPY main.py .
COPY ./monitoring ./monitoring
# Set Python path so local modules are visible
ENV PYTHONPATH=/app:/app/src

# Expose port for FastAPI
EXPOSE 5006

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "5006"]

