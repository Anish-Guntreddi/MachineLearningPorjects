"""
Home page - Project grid with category filters.
"""
import streamlit as st
from components.model_loader import discover_projects, get_category_style, format_metric_value


def render():
    # Hero
    st.markdown('<div class="hero-title">Machine Learning Portfolio</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">12 end-to-end deep learning projects spanning computer vision, '
                'natural language processing, audio, and multimodal systems.</div>', unsafe_allow_html=True)
    st.markdown("")

    projects = discover_projects()

    if not projects:
        st.warning("No projects found. Run `python generate_precomputed.py` first.")
        return

    # Category filter
    categories = sorted(set(p['card'].get('category', 'Other') for p in projects))
    all_categories = ["All"] + categories

    selected = st.pills("Filter by category", all_categories, default="All", key="cat_filter")
    if not selected:
        selected = "All"

    st.markdown("")

    # Filter projects
    filtered = projects if selected == "All" else [
        p for p in projects if p['card'].get('category') == selected
    ]

    # Render card grid (3 columns)
    cols = st.columns(3)
    for i, project in enumerate(filtered):
        card = project['card']
        results = project.get('results')
        category = card.get('category', 'Other')
        style = get_category_style(category)
        project_id = card.get('project_id', project['dir'][:2])

        # Get primary metric
        metric_value = ""
        metric_label = ""
        if results and 'metrics' in results:
            metrics = results['metrics']
            for key, val in metrics.items():
                metric_value = format_metric_value(val, key)
                metric_label = key.replace('test_', '').replace('_', ' ').title()
                break

        with cols[i % 3]:
            st.markdown(f"""
            <div class="project-card {style['css']} fade-in fade-in-{i + 1}">
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

            if st.button(f"View Details", key=f"home_card_{project_id}", use_container_width=True):
                st.session_state['selected_project'] = project_id
                st.session_state['page'] = 'Project Detail'
                st.rerun()
