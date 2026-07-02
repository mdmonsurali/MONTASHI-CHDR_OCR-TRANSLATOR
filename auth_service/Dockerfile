FROM python:3.11.15-slim-trixie

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY auth_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY auth_service/src /app/src
COPY database/schema.sql /workspace/database/schema.sql

WORKDIR /app/src

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
