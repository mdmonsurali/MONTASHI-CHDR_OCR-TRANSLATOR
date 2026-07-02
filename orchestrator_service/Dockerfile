# orchestrator_service
#
# Stateless coordinator. Drives the full upload → OCR → swap GPU → translate
# pipeline by calling ocr_service and translator_service over the docker
# network and stopping/starting their containers via the docker socket.
# No GPU needed in this image.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY orchestrator_service/requirements.txt /workspace/orchestrator_service/
RUN pip install --no-cache-dir -r /workspace/orchestrator_service/requirements.txt

COPY orchestrator_service/src/ /workspace/orchestrator_service/src/

ENV PYTHONPATH=/workspace:/workspace/orchestrator_service/src:$PYTHONPATH
ENV ORCHESTRATOR_SERVICE_PORT=8004
ENV OCR_SERVICE_URL=http://ocr_service:8001
ENV TRANSLATOR_SERVICE_URL=http://translator_service:8003
ENV GPU_SWAP=true
ENV OCR_CONTAINER_NAME=ocr_service
ENV TRANSLATOR_CONTAINER_NAME=translator_service
ENV SWAP_HEALTH_TIMEOUT_SEC=300
ENV TRANSLATE_TARGET_LANG=pt-BR

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:8004/health || exit 1

WORKDIR /workspace/orchestrator_service/src
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
