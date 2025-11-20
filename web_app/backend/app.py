"""
ML Portfolio Web Application - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any, List
import uvicorn
import os
from pathlib import Path

# Import our model registry
from models import MODEL_REGISTRY, get_model
from api.inference import router as inference_router
from api.metrics import router as metrics_router

# Initialize FastAPI app
app = FastAPI(
    title="ML Portfolio - Interactive Model Testing Platform",
    description="Test and explore 12 different machine learning models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include routers
app.include_router(inference_router, prefix="/api", tags=["inference"])
app.include_router(metrics_router, prefix="/api", tags=["metrics"])

# Root endpoint
@app.get("/")
async def root(request: Request):
    """
    Root endpoint - serves the main application page
    """
    return {
        "message": "ML Portfolio API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/api/docs",
            "models": "/api/models",
            "predict": "/api/{model_name}/predict",
            "metrics": "/api/{model_name}/metrics"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "models_loaded": len(MODEL_REGISTRY),
        "available_models": list(MODEL_REGISTRY.keys())
    }

# Get all available models
@app.get("/api/models")
async def get_models() -> Dict[str, Any]:
    """
    Get information about all available models
    """
    models_info = []

    for model_name, model_class in MODEL_REGISTRY.items():
        # Get model metadata
        model_info = {
            "id": model_name,
            "name": model_name,
            "display_name": model_name.replace("_", " ").title(),
            "description": model_class.__doc__ or f"{model_name} model",
            "category": _get_model_category(model_name),
            "status": "available",
            "input_type": _get_input_type(model_name),
            "icon": _get_model_icon(model_name),
            "github_url": f"https://github.com/yourusername/ml-portfolio/tree/main/{model_name.split('_')[0].zfill(2)}_{model_name}",
            "notebook_url": f"/notebooks/{model_name}.ipynb"
        }
        models_info.append(model_info)

    return {
        "total": len(models_info),
        "models": models_info
    }

def _get_model_category(model_name: str) -> str:
    """Get category for a model"""
    categories = {
        "image_classification": "Computer Vision",
        "object_detection": "Computer Vision",
        "instance_segmentation": "Computer Vision",
        "text_classification": "Natural Language Processing",
        "text_generation": "Natural Language Processing",
        "machine_translation": "Natural Language Processing",
        "speech_emotion_recognition": "Audio Processing",
        "automatic_speech_recognition": "Audio Processing",
        "recommender_system": "Recommender Systems",
        "time_series_forecasting": "Time Series",
        "anomaly_detection": "Anomaly Detection",
        "multimodal_fusion": "Multimodal Learning"
    }
    return categories.get(model_name, "Other")

def _get_input_type(model_name: str) -> str:
    """Get input type for a model"""
    input_types = {
        "image_classification": "image",
        "object_detection": "image",
        "instance_segmentation": "image",
        "text_classification": "text",
        "text_generation": "text",
        "machine_translation": "text",
        "speech_emotion_recognition": "audio",
        "automatic_speech_recognition": "audio",
        "recommender_system": "form",
        "time_series_forecasting": "csv",
        "anomaly_detection": "csv",
        "multimodal_fusion": "multimodal"
    }
    return input_types.get(model_name, "unknown")

def _get_model_icon(model_name: str) -> str:
    """Get emoji icon for a model"""
    icons = {
        "image_classification": "🖼️",
        "object_detection": "🎯",
        "instance_segmentation": "✂️",
        "text_classification": "📝",
        "text_generation": "✍️",
        "machine_translation": "🌐",
        "speech_emotion_recognition": "😊",
        "automatic_speech_recognition": "🎤",
        "recommender_system": "⭐",
        "time_series_forecasting": "📈",
        "anomaly_detection": "🔍",
        "multimodal_fusion": "🎭"
    }
    return icons.get(model_name, "🤖")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Run on application startup
    """
    print("=" * 70)
    print("ML Portfolio API Starting...")
    print("=" * 70)
    print(f"Available models: {len(MODEL_REGISTRY)}")
    for model_name in MODEL_REGISTRY.keys():
        print(f"  - {model_name}")
    print("=" * 70)
    print("API Documentation: http://localhost:8000/api/docs")
    print("=" * 70)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Run on application shutdown
    """
    print("\nShutting down ML Portfolio API...")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
