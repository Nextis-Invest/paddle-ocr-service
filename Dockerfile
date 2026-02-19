FROM python:3.10-slim

# System deps for PaddleOCR + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR models at build time (wget only — avoids segfault from running the engine in build env)
# Models are extracted into the exact paths PaddleOCR expects at runtime
ENV PADDLEOCR_HOME=/app/.paddleocr

RUN mkdir -p /app/.paddleocr/whl/det/en /app/.paddleocr/whl/rec/en /app/.paddleocr/whl/cls && \
    # Detection model (en_PP-OCRv3_det)
    wget -q -O /tmp/det.tar "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar" && \
    tar -xf /tmp/det.tar -C /app/.paddleocr/whl/det/en/ && \
    rm /tmp/det.tar && \
    # Recognition model (en_PP-OCRv4_rec)
    wget -q -O /tmp/rec.tar "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar" && \
    tar -xf /tmp/rec.tar -C /app/.paddleocr/whl/rec/en/ && \
    rm /tmp/rec.tar && \
    # Angle classification model
    wget -q -O /tmp/cls.tar "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar" && \
    tar -xf /tmp/cls.tar -C /app/.paddleocr/whl/cls/ && \
    rm /tmp/cls.tar

# Copy app
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
