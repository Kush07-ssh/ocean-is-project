"""
model_server.py — FastAPI wrapper around OceanModel.py
Runs in the model container, exposes POST /analyze
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCEAN Model Server", version="1.0")


class ScoresRequest(BaseModel):
    scores: dict  # e.g. {"Openness": 72.5, "Conscientiousness": 55.0, ...}


class AnalysisResponse(BaseModel):
    analysis: str


# Lazy-load on first request (model takes time to load)
_model_loaded = False


def ensure_model():
    global _model_loaded
    if not _model_loaded:
        logger.info("Loading model for the first time...")
        from OceanModel import _load_model

        _load_model()
        _model_loaded = True
        logger.info("Model loaded and ready.")


@app.on_event("startup")
async def startup_event():
    """Pre-warm the model on container startup so first request isn't slow."""
    logger.info("Pre-warming model on startup...")
    try:
        ensure_model()
    except Exception as e:
        logger.error(f"Model pre-warm failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_loaded}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: ScoresRequest):
    try:
        ensure_model()
        from OceanModel import llm_analysis

        result = llm_analysis(request.scores)
        return AnalysisResponse(analysis=result)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
