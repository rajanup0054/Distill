FROM python:3.11-slim

# Graphviz binary is needed to draw the mind map
RUN apt-get update && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False
CMD ["python", "app.py"]
