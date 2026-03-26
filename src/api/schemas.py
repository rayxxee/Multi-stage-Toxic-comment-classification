from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="A list of texts to classify.",
        min_items=1,
        example=["This is a safe comment.", "You are an idiot."]
    )

class PredictionResult(BaseModel):
    text: str = Field(description="The original input text.")
    label: str = Field(description="The predicted class label (e.g., 'safe', 'not_safe').")
    probability: Optional[float] = Field(None, description="The probability associated with the prediction, if applicable.")
    model_used: str = Field(description="The name of the pipeline stage that made the prediction.")

class PredictionResponse(BaseModel):
    results: List[PredictionResult] = Field(description="List of prediction results corresponding to the input texts.")
