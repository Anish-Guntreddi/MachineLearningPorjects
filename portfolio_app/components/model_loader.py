"""
Data discovery and model loading for portfolio app.
Scans repo root for XX_*/model_card.yaml and results.yaml.
"""
import yaml
import streamlit as st
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

CATEGORY_COLORS = {
    "Computer Vision": {"css": "vision", "color": "#2563eb", "bg": "#dbeafe"},
    "NLP": {"css": "nlp", "color": "#d97706", "bg": "#fef3c7"},
    "Audio": {"css": "audio", "color": "#7c3aed", "bg": "#ede9fe"},
    "Tabular": {"css": "tabular", "color": "#059669", "bg": "#d1fae5"},
    "Multimodal": {"css": "multimodal", "color": "#e11d48", "bg": "#ffe4e6"},
}

PLOTLY_COLORS = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#e11d48", "#0891b2", "#4f46e5", "#ca8a04"]


@st.cache_data
def discover_projects():
    """Scan repo root for project directories with model_card.yaml."""
    projects = []
    for d in sorted(REPO_ROOT.iterdir()):
        if d.is_dir() and d.name[:2].isdigit() and d.name[2] == '_':
            card_path = d / 'model_card.yaml'
            if not card_path.exists():
                continue
            try:
                card = yaml.safe_load(card_path.read_text(encoding='utf-8'))
            except Exception:
                continue
            results = None
            results_path = d / 'results.yaml'
            if results_path.exists():
                try:
                    results = yaml.safe_load(results_path.read_text(encoding='utf-8'))
                except Exception:
                    pass
            projects.append({
                'dir': d.name,
                'path': str(d),
                'card': card,
                'results': results,
            })
    return projects


def get_project_by_id(project_id):
    """Get a single project by its ID (e.g. '01')."""
    for p in discover_projects():
        if p['card'].get('project_id') == project_id:
            return p
    return None


def get_category_style(category):
    """Get styling info for a category."""
    return CATEGORY_COLORS.get(category, {"css": "vision", "color": "#2563eb", "bg": "#dbeafe"})


def format_metric_value(value, metric_name):
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    name_lower = metric_name.lower()
    if any(k in name_lower for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'map']):
        if isinstance(value, (int, float)) and value <= 1.0:
            return f"{value * 100:.1f}%"
        return f"{value:.1f}%"
    if any(k in name_lower for k in ['loss', 'rmse', 'mae', 'mse']):
        return f"{value:.4f}"
    if 'perplexity' in name_lower:
        return f"{value:.1f}"
    if 'bleu' in name_lower:
        return f"{value:.1f}"
    if 'wer' in name_lower:
        return f"{value:.1f}%"
    return f"{value}"


@st.cache_resource
def load_model_safe(project_dir, model_name):
    """Attempt to load a trained model. Returns dict with model and live status."""
    try:
        import sys
        import torch
        project_path = REPO_ROOT / project_dir
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))

        # Try loading checkpoint
        checkpoint_dir = project_path / 'checkpoints'
        if not checkpoint_dir.exists():
            return {"model": None, "live": False}

        best_model = checkpoint_dir / 'best_model.pth'
        if not best_model.exists():
            # Try any .pth file
            pth_files = list(checkpoint_dir.glob('*.pth'))
            if not pth_files:
                return {"model": None, "live": False}
            best_model = pth_files[0]

        checkpoint = torch.load(best_model, map_location='cpu', weights_only=False)
        return {"model": checkpoint, "live": True}
    except Exception:
        return {"model": None, "live": False}
