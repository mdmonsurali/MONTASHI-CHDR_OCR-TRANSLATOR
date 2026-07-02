FROM python:3.11.15-slim-trixie

ENV PYTHONUNBUFFERED=1 \
    OCR_SERVICE_URL=http://ocr_service:8001

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
