from typing import List, Dict
from src.api.schemas import PredictionResult
from src.pipeline.pipeline import HierarchicalPipeline
from src.data.cleaner import TextCleaner

def process_predictions(
    texts: List[str], 
    pipeline: HierarchicalPipeline, 
    cleaner: TextCleaner, 
    binary_labels: Dict[int, str]
) -> List[PredictionResult]:
    """
    Business logic layer for processing text predictions.
    Applies standard TextCleaner to incoming data before feeding
    it to the HierarchicalPipeline.
    """
    results = []
    
    # Preprocess all texts
    cleaned_texts = [cleaner.clean_text(doc) for doc in texts]
    
    for original_text, cleaned_text in zip(texts, cleaned_texts):
        # Handle cases where cleaning removes all characters (empty comments)
        if not cleaned_text.strip():
            # If the text is empty/noise, we can default it to benign/safe
            safe_label = binary_labels.get(1, "safe")
            results.append(PredictionResult(
                text=original_text,
                label=safe_label,
                probability=1.0,
                model_used="preprocessor_filter"
            ))
            continue
            
        # The pipeline accepts lists, so we pass it as a single-item list
        y_pred, per_stage_data = pipeline.run([cleaned_text], y_true=None)
        
        assigned_label_idx = y_pred[0]
        label_name = binary_labels.get(assigned_label_idx, str(assigned_label_idx))
        
        # Identify the stage that made this prediction by looking at populated subsets
        model_used_name = "unknown"
        for stage_name, data in per_stage_data.items():
            if len(data.get('y_pred_subset', [])) > 0:
                model_used_name = stage_name
                break
                
        results.append(PredictionResult(
            text=original_text,
            label=label_name,
            probability=None, # Pipeline obscures probability locally
            model_used=model_used_name
        ))
        
    return results
