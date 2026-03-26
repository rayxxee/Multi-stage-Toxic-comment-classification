from fastapi import FastAPI
from contextlib import asynccontextmanager
import yaml
import os
import joblib
import logging

from src.models.gatekeeper_lr import GatekeeperLR
# Ensure you have the fasttext model in the path
from src.pipeline.pipeline import HierarchicalPipeline
from src.data.cleaner import TextCleaner
from src.api.routes import router as prediction_router

logger = logging.getLogger("api_main")
logging.basicConfig(level=logging.INFO)

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing ML models and preprocessors...")
    try:
        config = load_config()
        models_dir = config['paths']['model_output']
        
        # Initialize Cleaner
        app.state.cleaner = TextCleaner(regex_pattern=config['preprocessing']['regex_pattern'])
        
        # Load Gatekeeper
        gatekeeper_path = os.path.join(models_dir, 'gatekeeper.joblib')
        if not os.path.exists(gatekeeper_path):
            raise FileNotFoundError(f"Gatekeeper model not found at {gatekeeper_path}")
        gatekeeper = GatekeeperLR.load(gatekeeper_path)
        
        # Load FastText
        fasttext_path = os.path.join(models_dir, 'fasttext.joblib')
        if not os.path.exists(fasttext_path):
            raise FileNotFoundError(f"Fasttext model not found at {fasttext_path}")
        fasttext_model = joblib.load(fasttext_path)
        
        available_stages = {
            'gatekeeper_lr': {'name': 'gatekeeper_lr', 'model': gatekeeper, 'threshold': config['thresholds']['SAFE_THRESHOLD'], 'type': 'binary', 'pass_label': 1},
            'fasttext': {'name': 'fasttext', 'model': fasttext_model, 'threshold': config['thresholds']['FASTTEXT_THRESHOLD'], 'type': 'binary', 'pass_label': None}
        }
        
        pipeline_stages = [available_stages['gatekeeper_lr'], available_stages['fasttext']]
        pipeline = HierarchicalPipeline(stages=pipeline_stages)
        
        app.state.pipeline = pipeline
        app.state.binary_labels = {int(k): v for k, v in config['labels']['binary'].items()}
        logger.info("ML components loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}")
        # Not stopping the app here so the errors can be handled gracefully via HTTP 503
        app.state.pipeline = None
        app.state.cleaner = None
        app.state.binary_labels = None
        
    yield
    # Shutdown
    logger.info("Shutting down ML models...")
    app.state.pipeline = None
    app.state.binary_labels = None
    app.state.cleaner = None

app = FastAPI(
    title="Model as a Service API",
    description="REST API for hierarchical text classification pipelines.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(prediction_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
