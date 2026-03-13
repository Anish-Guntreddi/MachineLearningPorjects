"""
Project Detail page - Tabbed layout with Overview, Metrics, Technical Details, Notebook, Try It, and Compare.
"""
import streamlit as st
from components.model_loader import (
    discover_projects, get_project_by_id, get_category_style, format_metric_value, PLOTLY_COLORS
)
from components.metrics_display import (
    training_curves, accuracy_curves, model_comparison_chart,
    per_class_heatmap, learning_rate_curve, model_comparison_table, radar_chart
)
from components.notebook_renderer import render_notebook


def render():
    projects = discover_projects()
    selected_id = st.session_state.get('selected_project', '01')

    project = get_project_by_id(selected_id)
    if not project:
        st.error(f"Project {selected_id} not found.")
        return

    card = project['card']
    results = project.get('results', {})
    category = card.get('category', 'Other')
    style = get_category_style(category)

    # Back button
    if st.button("< Back to All Projects", key="back_btn"):
        st.session_state['page'] = 'home'
        st.session_state['selected_project'] = None
        st.rerun()

    # Header
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <span class="badge badge-{style['css']}">{category}</span>
        <div class="hero-title" style="font-size: 2rem; margin-top: 8px;">{card.get('title', '')}</div>
        <div class="hero-subtitle">{card.get('short_description', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick info row
    info_cols = st.columns(4)
    dataset = card.get('dataset', {})
    with info_cols[0]:
        st.markdown(f"**Dataset:** {dataset.get('name', 'N/A')}")
    with info_cols[1]:
        st.markdown(f"**Best Model:** {results.get('best_model', card.get('default_model', 'N/A'))}")
    with info_cols[2]:
        st.markdown(f"**Input:** {card.get('input_type', 'N/A')}")
    with info_cols[3]:
        st.markdown(f"**Output:** {card.get('output_type', 'N/A')}")

    st.markdown("---")

    # ── Tabbed layout ──
    tab_overview, tab_metrics, tab_tech, tab_notebook, tab_demo, tab_compare = st.tabs([
        "Overview", "All Metrics", "Technical Details", "Notebook", "Try It", "Compare"
    ])

    with tab_overview:
        _render_overview(results, card)

    with tab_metrics:
        _render_all_metrics(results)

    with tab_tech:
        _render_technical_details(card, project)

    with tab_notebook:
        st.markdown("#### Jupyter Notebook")
        st.caption("Full notebook source code and outputs from the training run.")
        render_notebook(project['path'])

    with tab_demo:
        _render_try_it(project)

    with tab_compare:
        _render_compare(selected_id, projects)


    # Tags footer
    tags = card.get('tags', [])
    if tags:
        st.markdown("---")
        tag_html = " ".join(
            f'<span style="background:#1c1917;border:1px solid #292524;border-radius:2px;'
            f'padding:2px 8px;font-family:JetBrains Mono;font-size:0.75rem;color:#a8a29e;">{t}</span>'
            for t in tags
        )
        st.markdown(tag_html, unsafe_allow_html=True)


# ── Tab renderers ──────────────────────────────────────────

def _render_overview(results, card):
    """Overview tab: key metrics + training curves."""
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        if results and 'metrics' in results:
            st.markdown("#### Key Metrics")
            metrics = results['metrics']
            metric_cols = st.columns(min(len(metrics), 4))
            for i, (key, val) in enumerate(metrics.items()):
                with metric_cols[i % len(metric_cols)]:
                    label = key.replace('test_', '').replace('_', ' ').title()
                    st.metric(label, format_metric_value(val, key))

        history = results.get('training_history', {})
        if history:
            st.markdown("#### Training History")
            fig_loss = training_curves(history)
            if fig_loss:
                st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})
            fig_acc = accuracy_curves(history)
            if fig_acc:
                st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        comparisons = results.get('model_comparison', [])
        if comparisons:
            st.markdown("#### Model Comparison")
            fig_comp = model_comparison_chart(comparisons)
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

        optuna = results.get('optuna', {})
        if optuna:
            st.markdown("#### Hyperparameter Search")
            st.markdown(f"**Trials:** {optuna.get('n_trials', 'N/A')}")
            best_trial = optuna.get('best_trial', 'N/A')
            best_val = optuna.get('best_val_loss', optuna.get('best_accuracy', optuna.get('best_val_loss', 'N/A')))
            st.markdown(f"**Best trial:** #{best_trial}")
            if isinstance(best_val, (int, float)):
                st.markdown(f"**Best value:** `{best_val:.6f}`")
            best_params = optuna.get('best_params', {})
            if best_params:
                params_md = "\n".join(f"- **{k}:** `{v}`" for k, v in best_params.items())
                st.markdown(params_md)


def _render_all_metrics(results):
    """All Metrics tab: full table, per-class, LR schedule, model comparison table."""
    if not results:
        st.info("No results available for this project.")
        return

    metrics = results.get('metrics', {})
    if metrics:
        st.markdown("#### All Test Metrics")
        metric_cols = st.columns(min(len(metrics), 4))
        for i, (key, val) in enumerate(metrics.items()):
            with metric_cols[i % len(metric_cols)]:
                label = key.replace('test_', '').replace('_', ' ').title()
                st.metric(label, format_metric_value(val, key))

    comparisons = results.get('model_comparison', [])
    if comparisons:
        st.markdown("#### Model Architecture Comparison")
        table_html = model_comparison_table(comparisons)
        if table_html:
            st.markdown(table_html, unsafe_allow_html=True)

    per_class = results.get('per_class_metrics', [])
    if per_class:
        st.markdown("#### Per-Class Performance")
        fig_heatmap = per_class_heatmap(per_class)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})

    history = results.get('training_history', {})
    if history:
        fig_lr = learning_rate_curve(history)
        if fig_lr:
            st.markdown("#### Learning Rate Schedule")
            st.plotly_chart(fig_lr, use_container_width=True, config={'displayModeBar': False})

    final_training = results.get('final_training', {})
    if final_training:
        st.markdown("#### Training Summary")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Model", final_training.get('model_name', 'N/A'))
        with cols[1]:
            st.metric("Epochs Trained", final_training.get('epochs_trained', 'N/A'))
        with cols[2]:
            best_val = final_training.get('best_val_loss', final_training.get('best_val_accuracy', None))
            if best_val is not None:
                st.metric("Best Val Score", f"{best_val:.6f}")


def _render_technical_details(card, project):
    """Technical Details tab: technical overview, architecture, training config, methodology, dataset."""

    # Technical Overview — full ML pipeline breakdown
    overview = card.get('technical_overview', {})
    if overview:
        st.markdown("#### Technical Overview")
        overview_summary = overview.get('summary', '')
        if overview_summary:
            st.markdown(overview_summary)

        pipeline = overview.get('pipeline', [])
        if pipeline:
            st.markdown("")
            st.markdown("**ML Pipeline**")
            for i, step in enumerate(pipeline, 1):
                # Split on first colon to bold the step name
                if ':' in step:
                    name, desc = step.split(':', 1)
                    st.markdown(
                        f'<div style="margin:6px 0;padding:8px 12px;background:#1c1917;border-left:3px solid #2dd4bf;'
                        f'border-radius:2px;">'
                        f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#2dd4bf;">{i}.</span> '
                        f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#fafaf9;">{name.strip()}</span>'
                        f'<span style="font-family:Source Sans 3;font-size:0.85rem;color:#a8a29e;">:{desc}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="margin:6px 0;padding:8px 12px;background:#1c1917;border-left:3px solid #2dd4bf;'
                        f'border-radius:2px;">'
                        f'<span style="font-family:JetBrains Mono;font-size:0.8rem;color:#2dd4bf;">{i}.</span> '
                        f'<span style="font-family:Source Sans 3;font-size:0.85rem;color:#e7e5e4;">{step}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.markdown("---")

    arch = card.get('architecture', {})
    if arch:
        st.markdown("#### Architecture")
        cols = st.columns([1, 1])
        with cols[0]:
            st.markdown(f"**Type:** {arch.get('type', 'N/A')}")
            st.markdown(f"**Framework:** {arch.get('framework', 'PyTorch')}")
            pretrained = arch.get('pretrained', False)
            st.markdown(f"**Pretrained:** {'Yes' if pretrained else 'No (trained from scratch)'}")
        with cols[1]:
            components = arch.get('key_components', [])
            if components:
                st.markdown("**Key Components:**")
                for c in components:
                    st.markdown(f"- {c}")

    training = card.get('training', {})
    if training:
        st.markdown("---")
        st.markdown("#### Training Configuration")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Loss Function:** `{training.get('loss_function', 'N/A')}`")
            st.markdown(f"**Optimizer:** `{training.get('optimizer', 'N/A')}`")
            st.markdown(f"**Scheduler:** `{training.get('scheduler', 'N/A')}`")
        with cols[1]:
            st.markdown(f"**Mixed Precision:** {'Yes' if training.get('mixed_precision') else 'No'}")
            st.markdown(f"**Gradient Clipping:** {'Yes' if training.get('gradient_clipping') else 'No'}")
            augmentation = training.get('augmentation', [])
            if augmentation:
                st.markdown(f"**Augmentation:** {', '.join(augmentation)}")

    methodology = card.get('methodology', {})
    if methodology:
        st.markdown("---")
        st.markdown("#### Methodology")
        summary = methodology.get('summary', '')
        if summary:
            st.markdown(summary)
        techniques = methodology.get('key_techniques', [])
        if techniques:
            st.markdown("**Key Techniques:**")
            tech_html = " ".join(
                f'<span style="background:#042f2e;border:1px solid #0f766e;border-radius:2px;'
                f'padding:2px 8px;font-family:JetBrains Mono;font-size:0.75rem;color:#2dd4bf;'
                f'margin-right:4px;">{t}</span>'
                for t in techniques
            )
            st.markdown(tech_html, unsafe_allow_html=True)

    dataset = card.get('dataset', {})
    if dataset:
        st.markdown("---")
        st.markdown("#### Dataset")
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"**Name:** {dataset.get('name', 'N/A')}")
            st.markdown(f"**Source:** {dataset.get('source', 'N/A')}")
        with cols[1]:
            train_size = dataset.get('train_size')
            test_size = dataset.get('test_size')
            if train_size:
                st.markdown(f"**Train samples:** {train_size:,}")
            if test_size:
                st.markdown(f"**Test samples:** {test_size:,}")
        with cols[2]:
            num_classes = dataset.get('num_classes')
            if num_classes:
                st.markdown(f"**Classes:** {num_classes}")
            class_names = dataset.get('class_names', [])
            if class_names and len(class_names) <= 15:
                st.markdown(f"**Labels:** {', '.join(class_names)}")

    models = card.get('models_available', [])
    if models:
        st.markdown("---")
        st.markdown("#### Available Models")
        model_html = " ".join(
            f'<code style="background:#1c1917;padding:2px 6px;border-radius:2px;font-size:0.8rem;'
            f'color:#e7e5e4;">{m}</code>'
            for m in models
        )
        default = card.get('default_model', '')
        if default:
            st.markdown(f"**Default:** `{default}`")
        st.markdown(model_html, unsafe_allow_html=True)

    if not arch and not training and not methodology:
        _render_readme_fallback(project)


def _render_readme_fallback(project):
    """Fall back to rendering the project README if no technical details in YAML."""
    from pathlib import Path
    readme_path = Path(project['path']) / 'README.md'
    if readme_path.exists():
        try:
            content = readme_path.read_text(encoding='utf-8')
            st.markdown("---")
            st.markdown("#### Project Documentation (README)")
            lines = content.split('\n')
            if len(lines) > 500:
                st.markdown('\n'.join(lines[:500]))
                st.caption(f"Showing first 500 of {len(lines)} lines. See full README in the repository.")
            else:
                st.markdown(content)
        except Exception:
            pass


# ── Try It (Coming Soon) ──────────────────────────────────

def _render_try_it(project):
    """Coming soon placeholder for interactive demos."""
    st.markdown("#### Try It Out")
    st.markdown("")
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;background:#1c1917;border:1px solid #292524;'
        'border-radius:8px;margin:20px 0;">'
        '<div style="font-size:2rem;color:#2dd4bf;margin-bottom:12px;">Coming Soon</div>'
        '<div style="font-size:1rem;color:#a8a29e;max-width:500px;margin:0 auto;">'
        'Interactive model demos are on the way. You will be able to upload your own data '
        'and test each model directly in the browser.</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ── Compare Tab ───────────────────────────────────────────

def _render_compare(current_id, projects):
    """Compare this project's metrics with other projects."""
    st.markdown("#### Compare with Other Projects")
    st.caption("Select projects to compare side by side with the current one.")

    # Build options excluding current project
    options = {}
    for p in projects:
        pid = p['card'].get('project_id', p['dir'][:2])
        if pid != current_id:
            options[pid] = p['card'].get('title', p['dir'])

    compare_ids = st.multiselect(
        "Select projects to compare",
        options=list(options.keys()),
        format_func=lambda x: f"{x} - {options[x]}",
        default=[],
        max_selections=4,
        key="compare_select_inline",
    )

    if not compare_ids:
        st.info("Select one or more projects above to compare metrics.")
        return

    # Build full list: current project + selected
    all_ids = [current_id] + compare_ids
    selected_projects = [p for p in projects if p['card'].get('project_id') in all_ids]

    # Sort so current project is first
    selected_projects.sort(key=lambda p: 0 if p['card'].get('project_id') == current_id else 1)

    # Metric cards side by side
    st.markdown("---")
    st.markdown("#### Key Metrics")
    cols = st.columns(len(selected_projects))
    for i, proj in enumerate(selected_projects):
        pcard = proj['card']
        presults = proj.get('results', {})
        pstyle = get_category_style(pcard.get('category', 'Other'))
        is_current = pcard.get('project_id') == current_id

        with cols[i]:
            border = "border:2px solid #2dd4bf;" if is_current else ""
            st.markdown(f"""
            <div class="project-card {pstyle['css']}" style="text-align:center;{border}">
                <div class="card-number">PROJECT {pcard.get('project_id', '')}</div>
                <div class="card-title" style="font-size:0.95rem;">{pcard.get('title', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            metrics = presults.get('metrics', {})
            for key, val in metrics.items():
                label = key.replace('test_', '').replace('_', ' ').title()
                st.metric(label, format_metric_value(val, key))

    # Radar chart
    all_metrics = set()
    for proj in selected_projects:
        presults = proj.get('results', {})
        all_metrics.update(presults.get('metrics', {}).keys())

    metric_ranges = {}
    for metric in all_metrics:
        values = []
        for proj in selected_projects:
            val = proj.get('results', {}).get('metrics', {}).get(metric)
            if val is not None:
                values.append(float(val))
        if values:
            metric_ranges[metric] = (min(values), max(values))

    radar_data = {}
    for proj in selected_projects:
        pcard = proj['card']
        name = f"{pcard.get('project_id', '')} {pcard.get('title', '')}"
        metrics = proj.get('results', {}).get('metrics', {})
        normalized = {}
        for metric in all_metrics:
            val = metrics.get(metric)
            if val is not None and metric in metric_ranges:
                mn, mx = metric_ranges[metric]
                if mx > mn:
                    is_lower_better = any(k in metric.lower()
                                          for k in ['loss', 'error', 'rmse', 'mae', 'wer', 'perplexity'])
                    norm = (float(val) - mn) / (mx - mn) * 100
                    if is_lower_better:
                        norm = 100 - norm
                else:
                    norm = 50
                label = metric.replace('test_', '').replace('_', ' ').title()
                normalized[label] = round(norm, 1)
        if normalized:
            radar_data[name] = normalized

    if radar_data and len(radar_data) >= 2:
        st.markdown("#### Normalized Comparison")
        fig = radar_chart(radar_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Comparison table
    st.markdown("#### Detailed Comparison")
    headers = ["Metric"] + [
        f"{p['card'].get('project_id', '')} {p['card'].get('title', '')}" for p in selected_projects
    ]
    table_data = []
    for metric in sorted(all_metrics):
        row = [metric.replace('test_', '').replace('_', ' ').title()]
        for proj in selected_projects:
            val = proj.get('results', {}).get('metrics', {}).get(metric)
            row.append(format_metric_value(val, metric) if val is not None else "-")
        table_data.append(row)

    header_html = "".join(f'<th style="padding:8px 12px;text-align:left;font-family:JetBrains Mono;'
                          f'font-size:0.8rem;border-bottom:2px solid #2dd4bf;color:#fafaf9;">{h}</th>'
                          for h in headers)
    rows_html = ""
    for row in table_data:
        cells = "".join(
            f'<td style="padding:8px 12px;font-family:{("JetBrains Mono" if i > 0 else "Source Sans 3")};'
            f'font-size:0.85rem;border-bottom:1px solid #292524;color:#e7e5e4;">{c}</td>'
            for i, c in enumerate(row)
        )
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;margin-top:16px;">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
