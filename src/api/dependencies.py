from fastapi import Request, HTTPException
from src.pipeline.pipeline import HierarchicalPipeline
from src.data.cleaner import TextCleaner
from typing import Dict

def get_pipeline(request: Request) -> HierarchicalPipeline:
    pipeline = getattr(request.app.state, 'pipeline', None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded or initializing.")
    return pipeline

def get_cleaner(request: Request) -> TextCleaner:
    cleaner = getattr(request.app.state, 'cleaner', None)
    if not cleaner:
        raise HTTPException(status_code=503, detail="TextCleaner not initialized.")
    return cleaner

def get_binary_labels(request: Request) -> Dict[int, str]:
    labels = getattr(request.app.state, 'binary_labels', None)
    if not labels:
        raise HTTPException(status_code=503, detail="Labels configuration not loaded.")
    return labels
