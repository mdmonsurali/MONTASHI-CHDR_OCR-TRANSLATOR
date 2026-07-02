FROM vllm/vllm-openai:unlimited-ocr

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libreoffice \
    poppler-utils \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    imagemagick \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# ImageMagick's default policy on Debian/Ubuntu blocks PDF reads (Ghostscript
# CVE mitigation). Our formula renderer generates the PDF itself via pdflatex
# and immediately converts it to PNG — there's no untrusted PDF input — so
# this is safe to re-enable. Without it the matplotlib mathtext fallback
# fires, which can't parse `\frac{\text{...}}{...}` or empty args.
RUN sed -i 's|<policy domain="coder" rights="none" pattern="PDF"|<policy domain="coder" rights="read\|write" pattern="PDF"|g' \
    /etc/ImageMagick-6/policy.xml 2>/dev/null || true

COPY ocr_service/requirements.txt /workspace/ocr_service/
RUN pip3 install --no-cache-dir -r /workspace/ocr_service/requirements.txt

COPY ocr_service/src/ /workspace/ocr_service/src/
COPY ocr_service/entrypoint.sh /workspace/ocr_service/entrypoint.sh
RUN chmod +x /workspace/ocr_service/entrypoint.sh

# ocr_service's own copy of the reconstruction renderer modules. The
# translator_service ships an independent copy under translation_reconstruction
# so the two services can diverge without affecting each other.
COPY reconstruction_service/src/ocr_reconstruction /workspace/ocr_reconstruction

# Database schema bundled into the image; applied at startup.
COPY database/schema.sql /workspace/database/schema.sql

ENV PYTHONPATH=/workspace:/workspace/ocr_service/src:$PYTHONPATH
ENV OCR_SERVICE_PORT=8001
ENV VLLM_PORT=8888
ENV VLLM_SERVICE_URL=http://localhost:8888
# Served-model name baked into the vllm/vllm-openai:unlimited-ocr image.
ENV OCR_MODEL_NAME=Unlimited-OCR

EXPOSE 8888 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=180s \
    CMD curl -f http://localhost:8001/health || exit 1

WORKDIR /workspace/ocr_service/src

# entrypoint.sh probes the GPU and exports VLLM_MAX_NUM_SEQS for the right tier.
# All other knobs (GPU_MEM_UTIL, MAX_MODEL_LEN, OCR_MAX_TOKENS, VLLM_PORT,
# OCR_SERVICE_PORT) come from .env via docker-compose.
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["\
     source /workspace/ocr_service/entrypoint.sh; \
     vllm serve baidu/Unlimited-OCR \
         --tensor-parallel-size 1 \
         --gpu-memory-utilization $GPU_MEM_UTIL \
         --max-model-len $MAX_MODEL_LEN \
         --max-num-seqs $VLLM_MAX_NUM_SEQS \
         --enable-prefix-caching \
         --chat-template-content-format string \
         --served-model-name $OCR_MODEL_NAME \
         --trust-remote-code \
         --port $VLLM_PORT & \
     VLLM_PID=$!; \
     echo \"Waiting for vLLM (PID $VLLM_PID) to become ready on :$VLLM_PORT...\"; \
     for i in $(seq 1 300); do \
         if ! kill -0 $VLLM_PID 2>/dev/null; then \
             echo \"vLLM process died before becoming ready\" >&2; \
             exit 1; \
         fi; \
         if curl -sf http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then \
             echo \"vLLM ready after ${i}s\"; \
             break; \
         fi; \
         sleep 1; \
     done; \
     if ! curl -sf http://localhost:$VLLM_PORT/health >/dev/null 2>&1; then \
         echo \"vLLM did not become ready within 300s\" >&2; \
         exit 1; \
     fi; \
     exec uvicorn main:app --host 0.0.0.0 --port $OCR_SERVICE_PORT"]
