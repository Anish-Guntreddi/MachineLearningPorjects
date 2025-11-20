"""
Inference API Endpoints
Handles model predictions for all model types
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import time
from pathlib import Path

from models import get_model, MODEL_REGISTRY

router = APIRouter()

# Model cache to avoid reloading models
model_cache = {}

def get_cached_model(model_name: str):
    """
    Get model from cache or load it

    Args:
        model_name: Name of the model

    Returns:
        Loaded model instance
    """
    if model_name not in model_cache:
        model = get_model(model_name)
        model.ensure_loaded()
        model_cache[model_name] = model

    return model_cache[model_name]


@router.post("/{model_name}/predict")
async def predict(
    model_name: str,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Run prediction on uploaded data

    Args:
        model_name: Name of the model to use
        file: Uploaded file (for image/audio models)
        text: Text input (for NLP models)

    Returns:
        Prediction results
    """
    # Validate model exists
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Validate input
    if file is None and text is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' or 'text' input required"
        )

    try:
        # Get model
        model = get_cached_model(model_name)

        # Prepare input based on model type
        start_time = time.time()

        if file is not None:
            # Handle file upload (image, audio, etc.)
            contents = await file.read()

            # Run prediction
            result = model(contents)

        elif text is not None:
            # Handle text input
            result = model(text)

        else:
            raise HTTPException(status_code=400, detail="Invalid input")

        # Calculate processing time
        processing_time = time.time() - start_time

        # Add metadata
        response = {
            "status": "success",
            "model": model_name,
            "result": result,
            "processing_time_ms": round(processing_time * 1000, 2),
            "filename": file.filename if file else None
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/{model_name}/info")
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model

    Args:
        model_name: Name of the model

    Returns:
        Model information and metadata
    """
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    try:
        model = get_cached_model(model_name)
        info = model.get_model_info()

        return {
            "status": "success",
            "model_name": model_name,
            "info": info
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.post("/{model_name}/batch")
async def batch_predict(
    model_name: str,
    files: list[UploadFile] = File(...)
) -> Dict[str, Any]:
    """
    Run batch predictions on multiple files

    Args:
        model_name: Name of the model
        files: List of uploaded files

    Returns:
        List of prediction results
    """
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    # Limit batch size
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files allowed per batch"
        )

    try:
        model = get_cached_model(model_name)

        results = []
        start_time = time.time()

        for file in files:
            contents = await file.read()
            result = model(contents)

            results.append({
                "filename": file.filename,
                "prediction": result
            })

        total_time = time.time() - start_time

        return {
            "status": "success",
            "model": model_name,
            "total_files": len(files),
            "results": results,
            "total_processing_time_ms": round(total_time * 1000, 2),
            "avg_time_per_file_ms": round((total_time / len(files)) * 1000, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.delete("/{model_name}/cache")
async def clear_model_cache(model_name: str) -> Dict[str, Any]:
    """
    Clear cached model from memory

    Args:
        model_name: Name of the model to uncache

    Returns:
        Success message
    """
    if model_name in model_cache:
        del model_cache[model_name]
        return {
            "status": "success",
            "message": f"Model '{model_name}' removed from cache"
        }
    else:
        return {
            "status": "info",
            "message": f"Model '{model_name}' not in cache"
        }


@router.get("/cache/status")
async def get_cache_status() -> Dict[str, Any]:
    """
    Get status of model cache

    Returns:
        Information about cached models
    """
    return {
        "total_cached": len(model_cache),
        "cached_models": list(model_cache.keys()),
        "available_models": list(MODEL_REGISTRY.keys())
    }
