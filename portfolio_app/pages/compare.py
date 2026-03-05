"""
Compare page - Cross-project comparison with radar chart and table.
"""
import streamlit as st
from components.model_loader import discover_projects, get_category_style, format_metric_value, PLOTLY_COLORS
from components.metrics_display import radar_chart


def render():
    st.markdown('<div class="hero-title" style="font-size: 1.8rem;">Compare Projects</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Select 2-4 projects to compare metrics side by side.</div>',
                unsafe_allow_html=True)
    st.markdown("")

    projects = discover_projects()

    if not projects:
        st.warning("No projects found.")
        return

    # Multi-select
    options = {p['card'].get('project_id', p['dir'][:2]): p['card'].get('title', p['dir']) for p in projects}
    selected_ids = st.multiselect(
        "Select projects to compare",
        options=list(options.keys()),
        format_func=lambda x: f"{x} - {options[x]}",
        default=list(options.keys())[:3],
        max_selections=4,
        key="compare_select",
    )

    if len(selected_ids) < 2:
        st.info("Select at least 2 projects to compare.")
        return

    selected_projects = [p for p in projects if p['card'].get('project_id') in selected_ids]

    # Metric cards side by side
    st.markdown("### Key Metrics")
    cols = st.columns(len(selected_projects))
    for i, proj in enumerate(selected_projects):
        card = proj['card']
        results = proj.get('results', {})
        style = get_category_style(card.get('category', 'Other'))

        with cols[i]:
            st.markdown(f"""
            <div class="project-card {style['css']}" style="text-align: center;">
                <div class="card-number">PROJECT {card.get('project_id', '')}</div>
                <div class="card-title" style="font-size: 0.95rem;">{card.get('title', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            metrics = results.get('metrics', {})
            for key, val in metrics.items():
                label = key.replace('test_', '').replace('_', ' ').title()
                st.metric(label, format_metric_value(val, key))

    # Radar chart
    st.markdown("### Normalized Comparison")
    radar_data = {}
    all_metrics = set()

    for proj in selected_projects:
        results = proj.get('results', {})
        metrics = results.get('metrics', {})
        all_metrics.update(metrics.keys())

    # Normalize metrics to 0-100 scale
    metric_ranges = {}
    for metric in all_metrics:
        values = []
        for proj in selected_projects:
            val = proj.get('results', {}).get('metrics', {}).get(metric)
            if val is not None:
                values.append(float(val))
        if values:
            metric_ranges[metric] = (min(values), max(values))

    for proj in selected_projects:
        card = proj['card']
        name = f"{card.get('project_id', '')} {card.get('title', '')}"
        results = proj.get('results', {})
        metrics = results.get('metrics', {})

        normalized = {}
        for metric in all_metrics:
            val = metrics.get(metric)
            if val is not None and metric in metric_ranges:
                mn, mx = metric_ranges[metric]
                if mx > mn:
                    # For loss/error metrics, invert (lower is better)
                    is_lower_better = any(k in metric.lower() for k in ['loss', 'error', 'rmse', 'mae', 'wer', 'perplexity'])
                    norm = (float(val) - mn) / (mx - mn) * 100
                    if is_lower_better:
                        norm = 100 - norm
                else:
                    norm = 50
                label = metric.replace('test_', '').replace('_', ' ').title()
                normalized[label] = round(norm, 1)

        if normalized:
            radar_data[name] = normalized

    if radar_data:
        fig = radar_chart(radar_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Comparison table
    st.markdown("### Detailed Comparison")

    table_data = []
    headers = ["Metric"] + [
        f"{p['card'].get('project_id', '')} {p['card'].get('title', '')}" for p in selected_projects
    ]

    for metric in sorted(all_metrics):
        row = [metric.replace('test_', '').replace('_', ' ').title()]
        for proj in selected_projects:
            val = proj.get('results', {}).get('metrics', {}).get(metric)
            row.append(format_metric_value(val, metric) if val is not None else "-")
        table_data.append(row)

    # Render as HTML table
    header_html = "".join(f'<th style="padding:8px 12px;text-align:left;font-family:JetBrains Mono;'
                          f'font-size:0.8rem;border-bottom:2px solid #e7e5e4;">{h}</th>' for h in headers)
    rows_html = ""
    for row in table_data:
        cells = "".join(
            f'<td style="padding:8px 12px;font-family:{("JetBrains Mono" if i > 0 else "Source Sans 3")};'
            f'font-size:0.85rem;border-bottom:1px solid #e7e5e4;">{c}</td>'
            for i, c in enumerate(row)
        )
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;margin-top:16px;">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Model details
    st.markdown("### Model Details")
    for proj in selected_projects:
        card = proj['card']
        results = proj.get('results', {})
        style = get_category_style(card.get('category', 'Other'))

        comparisons = results.get('model_comparison', [])
        if comparisons:
            st.markdown(f"**{card.get('project_id', '')} - {card.get('title', '')}**")
            comp_cols = st.columns(len(comparisons))
            for j, comp in enumerate(comparisons):
                with comp_cols[j]:
                    st.markdown(f"**{comp.get('model', 'N/A')}**")
                    for k, v in comp.items():
                        if k != 'model':
                            st.caption(f"{k}: {v}")
