import base64
import io
import logging
import os
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PaddleOCR once at startup (downloads models if needed)
# use_angle_cls=True handles rotated text, lang='fr' for French docs
ocr_engine = None

@app.on_event("startup")
async def startup():
    global ocr_engine
    logger.info("Initializing PaddleOCR...")
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang="fr",
        use_gpu=False,
        show_log=False,
        enable_mkldnn=False,  # more stable on CPU-only servers
    )
    logger.info("PaddleOCR ready")


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
    return {"status": "ok", "engine": "paddleocr", "ready": ocr_engine is not None}


@app.post("/process-base64", response_model=OcrResponse)
async def process_base64(req: OcrRequest):
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
