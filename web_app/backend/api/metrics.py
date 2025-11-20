"""
Metrics API Endpoints
Handles model evaluation metrics and performance data
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pathlib import Path
import json

from models import MODEL_REGISTRY, get_model

router = APIRouter()


@router.get("/{model_name}/metrics")
async def get_model_metrics(model_name: str) -> Dict[str, Any]:
    """
    Get evaluation metrics for a specific model

    Args:
        model_name: Name of the model

    Returns:
        Model evaluation metrics
    """
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    try:
        model = get_model(model_name)
        metrics = model.get_metrics()

        if "error" in metrics:
            return {
                "status": "not_available",
                "model": model_name,
                "message": metrics["error"],
                "metrics": None
            }

        return {
            "status": "success",
            "model": model_name,
            "metrics": metrics
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load metrics: {str(e)}"
        )


@router.get("/summary")
async def get_all_metrics_summary() -> Dict[str, Any]:
    """
    Get summary of metrics for all models

    Returns:
        Summary of all model metrics
    """
    summary = []

    for model_name in MODEL_REGISTRY.keys():
        try:
            model = get_model(model_name)
            metrics = model.get_metrics()

            if "error" not in metrics:
                # Extract key metrics based on model type
                key_metrics = _extract_key_metrics(metrics)

                summary.append({
                    "model_name": model_name,
                    "status": "available",
                    **key_metrics
                })
            else:
                summary.append({
                    "model_name": model_name,
                    "status": "metrics_unavailable"
                })

        except Exception as e:
            summary.append({
                "model_name": model_name,
                "status": "error",
                "error": str(e)
            })

    return {
        "total_models": len(summary),
        "models": summary
    }


def _extract_key_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from full metrics dictionary

    Args:
        metrics: Full metrics dictionary

    Returns:
        Dictionary with key metrics only
    """
    key_metrics = {}

    # Try to extract common metrics
    if "final_metrics" in metrics:
        final = metrics["final_metrics"]
        if "accuracy" in final:
            key_metrics["accuracy"] = final["accuracy"]
        if "f1_score" in final:
            key_metrics["f1_score"] = final["f1_score"]
        if "precision" in final:
            key_metrics["precision"] = final["precision"]
        if "recall" in final:
            key_metrics["recall"] = final["recall"]

    # Model info
    if "model_name" in metrics:
        key_metrics["model_architecture"] = metrics["model_name"]
    if "dataset" in metrics:
        key_metrics["dataset"] = metrics["dataset"]
    if "total_parameters" in metrics:
        key_metrics["parameters"] = metrics["total_parameters"]
    if "training_time_minutes" in metrics:
        key_metrics["training_time_minutes"] = metrics["training_time_minutes"]

    return key_metrics


@router.get("/{model_name}/visualizations")
async def get_model_visualizations(model_name: str) -> Dict[str, Any]:
    """
    Get paths to model visualization files

    Args:
        model_name: Name of the model

    Returns:
        Paths to visualization files
    """
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    # Get project directory
    base_dir = Path(__file__).parent.parent.parent.parent
    project_mapping = {
        "image_classification": "01_Image_Classification",
        "object_detection": "02_Object_Detection",
        "text_classification": "04_Text_Classification",
        # Add more as needed
    }

    project_dir = project_mapping.get(model_name)
    if not project_dir:
        return {
            "status": "not_available",
            "message": f"Visualization mapping not configured for {model_name}"
        }

    results_dir = base_dir / project_dir / "results"

    if not results_dir.exists():
        return {
            "status": "not_available",
            "message": f"Results directory not found: {results_dir}"
        }

    # Find all PNG files in results directory
    visualizations = []
    for viz_file in results_dir.glob("*.png"):
        visualizations.append({
            "name": viz_file.stem,
            "filename": viz_file.name,
            "path": f"/static/results/{project_dir}/{viz_file.name}",
            "type": "image/png"
        })

    return {
        "status": "success",
        "model": model_name,
        "total_visualizations": len(visualizations),
        "visualizations": visualizations
    }


@router.get("/{model_name}/training-history")
async def get_training_history(model_name: str) -> Dict[str, Any]:
    """
    Get training history data for plotting

    Args:
        model_name: Name of the model

    Returns:
        Training history data
    """
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    try:
        model = get_model(model_name)
        metrics = model.get_metrics()

        if "error" in metrics:
            raise HTTPException(
                status_code=404,
                detail="Training history not available"
            )

        if "training_history" not in metrics:
            raise HTTPException(
                status_code=404,
                detail="Training history not found in metrics"
            )

        history = metrics["training_history"]

        return {
            "status": "success",
            "model": model_name,
            "history": history
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load training history: {str(e)}"
        )
