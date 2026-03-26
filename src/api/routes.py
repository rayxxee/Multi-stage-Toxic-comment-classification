from fastapi import APIRouter, Depends, HTTPException
import logging
from typing import Dict
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.dependencies import get_pipeline, get_cleaner, get_binary_labels
from src.api.services import process_predictions
from src.pipeline.pipeline import HierarchicalPipeline
from src.data.cleaner import TextCleaner

logger = logging.getLogger('api_router')
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    pipeline: HierarchicalPipeline = Depends(get_pipeline),
    cleaner: TextCleaner = Depends(get_cleaner),
    binary_labels: Dict[int, str] = Depends(get_binary_labels)
):
    try:
        results = process_predictions(
            texts=payload.texts,
            pipeline=pipeline,
            cleaner=cleaner,
            binary_labels=binary_labels
        )
        return PredictionResponse(results=results)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")
