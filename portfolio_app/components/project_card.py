"""
Reusable project card component with category accent.
"""
import streamlit as st
from components.model_loader import get_category_style, format_metric_value


def render_project_card(project, index=0):
    """Render a single project card. Returns True if clicked."""
    card = project['card']
    results = project.get('results')
    category = card.get('category', 'Other')
    style = get_category_style(category)

    # Get primary metric
    metric_value = ""
    metric_label = ""
    if results and 'metrics' in results:
        metrics = results['metrics']
        # Find the primary metric
        primary = card.get('evaluation', {}).get('primary_metric', '')
        for key, val in metrics.items():
            if primary and primary in key:
                metric_value = format_metric_value(val, key)
                metric_label = key.replace('test_', '').replace('_', ' ').title()
                break
        if not metric_value:
            # Use first metric
            key, val = next(iter(metrics.items()))
            metric_value = format_metric_value(val, key)
            metric_label = key.replace('test_', '').replace('_', ' ').title()

    project_id = card.get('project_id', project['dir'][:2])

    st.markdown(f"""
    <div class="project-card {style['css']} fade-in fade-in-{index + 1}" onclick="void(0)">
        <div class="card-number">PROJECT {project_id}</div>
        <div class="card-title">{card.get('title', project['dir'])}</div>
        <div class="card-description">{card.get('short_description', '')}</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <div class="card-metric">{metric_value}</div>
                <div class="card-metric-label">{metric_label}</div>
            </div>
            <span class="badge badge-{style['css']}">{category}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.button(f"View Project {project_id}", key=f"card_{project_id}", use_container_width=True)
