import base64
import io
import logging
import os
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

# Internal service only — no public CORS needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # closed — internal service, not called from browser
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# API Key auth — loaded from env var PADDLE_OCR_API_KEY
API_KEY = os.environ.get("PADDLE_OCR_API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: Optional[str] = Security(api_key_header)) -> str:
    if not API_KEY:
        # If no key configured, allow all (dev mode)
        logger.warning("PADDLE_OCR_API_KEY not set — running without auth (dev mode)")
        return ""
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# Model paths — baked into the Docker image at build time
# Dockerfile downloads to /app/.paddleocr/; we pass explicit paths so
# PaddleOCR doesn't try to re-download from the internet at runtime.
_PADDLE_HOME = "/app/.paddleocr/whl"
_DET_MODEL_DIR = f"{_PADDLE_HOME}/det/en/en_PP-OCRv3_det_infer"
_REC_MODEL_DIR = f"{_PADDLE_HOME}/rec/en/en_PP-OCRv4_rec_infer"
_CLS_MODEL_DIR = f"{_PADDLE_HOME}/cls/ch_ppocr_mobile_v2.0_cls_infer"

# Initialize PaddleOCR once at startup (models already baked in image)
ocr_engine = None


@app.on_event("startup")
async def startup():
    global ocr_engine
    logger.info("Initializing PaddleOCR with pre-downloaded models...")
    logger.info(f"  det: {_DET_MODEL_DIR}")
    logger.info(f"  rec: {_REC_MODEL_DIR}")
    logger.info(f"  cls: {_CLS_MODEL_DIR}")
    # lang='en' works perfectly for French/European Latin text
    # lang='fr' can segfault on some server configs (AVX issues)
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
        enable_mkldnn=False,  # more stable on CPU-only servers
        det_model_dir=_DET_MODEL_DIR,
        rec_model_dir=_REC_MODEL_DIR,
        cls_model_dir=_CLS_MODEL_DIR,
    )
    logger.info("✅ PaddleOCR ready (no runtime downloads)")


class OcrRequest(BaseModel):
    image_base64: str
    mime_type: Optional[str] = "image/jpeg"
    document_type: Optional[str] = None


class OcrResponse(BaseModel):
    success: bool
    text: str
    lines: list
    confidence: float
    processing_time_ms: int
    error: Optional[str] = None


@app.get("/health")
def health():
    models_present = all(
        os.path.isdir(p) for p in [_DET_MODEL_DIR, _REC_MODEL_DIR, _CLS_MODEL_DIR]
    )
    return {
        "status": "ok",
        "engine": "paddleocr",
        "ready": ocr_engine is not None,
        "models_baked": models_present,
        "model_paths": {
            "det": _DET_MODEL_DIR,
            "rec": _REC_MODEL_DIR,
            "cls": _CLS_MODEL_DIR,
        },
    }


@app.post("/process-base64", response_model=OcrResponse)
async def process_base64(req: OcrRequest, _key: str = Security(verify_api_key)):
    if ocr_engine is None:
        raise HTTPException(status_code=503, detail="OCR engine not ready")

    start = time.time()

    try:
        # Decode base64 image
        raw = req.image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        image_bytes = base64.b64decode(raw)

        # Convert to numpy array via PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        # Run OCR
        result = ocr_engine.ocr(img_array, cls=True)

        lines = []
        full_text_parts = []
        total_conf = 0.0
        count = 0

        if result and result[0]:
            for line in result[0]:
                box, (text, conf) = line
                lines.append({"text": text, "confidence": round(conf, 4), "box": box})
                full_text_parts.append(text)
                total_conf += conf
                count += 1

        avg_conf = round(total_conf / count, 4) if count > 0 else 0.0
        full_text = "\n".join(full_text_parts)
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(f"OCR done: {count} lines, conf={avg_conf}, {elapsed_ms}ms")

        return OcrResponse(
            success=True,
            text=full_text,
            lines=lines,
            confidence=avg_conf,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"OCR error: {e}")
        return OcrResponse(
            success=False,
            text="",
            lines=[],
            confidence=0.0,
            processing_time_ms=int((time.time() - start) * 1000),
            error=str(e),
        )
