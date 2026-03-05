"""
Project Detail page - Metrics, charts, training curves, and demo.
"""
import streamlit as st
from components.model_loader import (
    discover_projects, get_project_by_id, get_category_style, format_metric_value
)
from components.metrics_display import (
    training_curves, accuracy_curves, model_comparison_chart, per_class_heatmap
)
from components.demo_inputs import render_precomputed_demo


def render():
    projects = discover_projects()

    # Project selector
    project_ids = {p['card'].get('project_id', p['dir'][:2]): p['card'].get('title', p['dir']) for p in projects}

    selected_id = st.session_state.get('selected_project', '01')

    with st.sidebar:
        st.markdown("---")
        selected_id = st.selectbox(
            "Select Project",
            options=list(project_ids.keys()),
            format_func=lambda x: f"{x} - {project_ids[x]}",
            index=list(project_ids.keys()).index(selected_id) if selected_id in project_ids else 0,
            key="project_selector",
        )
        st.session_state['selected_project'] = selected_id

    project = get_project_by_id(selected_id)
    if not project:
        st.error(f"Project {selected_id} not found.")
        return

    card = project['card']
    results = project.get('results', {})
    category = card.get('category', 'Other')
    style = get_category_style(category)

    # Header
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("< All Projects"):
            st.session_state['page'] = 'Home'
            st.rerun()

    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <span class="badge badge-{style['css']}">{category}</span>
        <div class="hero-title" style="font-size: 2rem; margin-top: 8px;">{card.get('title', '')}</div>
        <div class="hero-subtitle">{card.get('short_description', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick info
    info_cols = st.columns(4)
    dataset = card.get('dataset', {})
    with info_cols[0]:
        st.markdown(f"**Dataset:** {dataset.get('name', 'N/A')}")
    with info_cols[1]:
        st.markdown(f"**Best Model:** {results.get('best_model', 'N/A')}")
    with info_cols[2]:
        st.markdown(f"**Input:** {card.get('input_type', 'N/A')}")
    with info_cols[3]:
        st.markdown(f"**Output:** {card.get('output_type', 'N/A')}")

    st.markdown("---")

    # Two-column layout: metrics/charts | demo
    left, right = st.columns([1.2, 1])

    with left:
        # Metrics
        if results and 'metrics' in results:
            st.markdown("#### Metrics")
            metrics = results['metrics']
            metric_cols = st.columns(min(len(metrics), 4))
            for i, (key, val) in enumerate(metrics.items()):
                with metric_cols[i % len(metric_cols)]:
                    label = key.replace('test_', '').replace('_', ' ').title()
                    st.metric(label, format_metric_value(val, key))

        # Training curves
        history = results.get('training_history', {})
        if history:
            st.markdown("#### Training History")
            fig_loss = training_curves(history)
            if fig_loss:
                st.plotly_chart(fig_loss, use_container_width=True, config={'displayModeBar': False})

            fig_acc = accuracy_curves(history)
            if fig_acc:
                st.plotly_chart(fig_acc, use_container_width=True, config={'displayModeBar': False})

        # Model comparison
        comparisons = results.get('model_comparison', [])
        if comparisons:
            st.markdown("#### Model Comparison")
            fig_comp = model_comparison_chart(comparisons)
            if fig_comp:
                st.plotly_chart(fig_comp, use_container_width=True, config={'displayModeBar': False})

        # Per-class metrics
        per_class = results.get('per_class_metrics', [])
        if per_class:
            st.markdown("#### Per-Class Performance")
            fig_heatmap = per_class_heatmap(per_class)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True, config={'displayModeBar': False})

    with right:
        st.markdown("#### Demo")
        demo_type = card.get('demo_type', 'precomputed')

        if demo_type == 'interactive':
            _render_interactive_demo(project)
        else:
            render_precomputed_demo(project)

    # Tags
    tags = card.get('tags', [])
    if tags:
        st.markdown("---")
        tag_html = " ".join(
            f'<span style="background:#f5f5f4;border:1px solid #e7e5e4;border-radius:2px;'
            f'padding:2px 8px;font-family:JetBrains Mono;font-size:0.75rem;color:#57534e;">{t}</span>'
            for t in tags
        )
        st.markdown(tag_html, unsafe_allow_html=True)


def _render_interactive_demo(project):
    """Route to the appropriate interactive demo based on project ID."""
    card = project['card']
    project_id = card.get('project_id', '')
    category = card.get('category', '')

    if project_id == '01':
        _demo_image_classification(project)
    elif project_id == '04':
        _demo_text_classification(project)
    elif project_id == '05':
        _demo_text_generation(project)
    elif project_id == '09':
        _demo_recommender(project)
    elif project_id == '10':
        _demo_time_series(project)
    elif project_id == '11':
        _demo_anomaly_detection(project)
    else:
        render_precomputed_demo(project)


def _demo_image_classification(project):
    """Interactive image classification demo."""
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="demo_img_cls")
    if uploaded:
        from PIL import Image
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.markdown("**Prediction (demo mode):**")
        # Without a trained model, show precomputed fallback
        results = project.get('results', {})
        samples = results.get('sample_predictions', [])
        if samples:
            st.info(f"Live inference requires a trained model. Showing sample predictions below.")
            render_precomputed_demo(project)
        else:
            st.info("Upload an image and train the model to see live predictions. "
                    "See metrics above for model performance.")
    else:
        st.caption("Upload a PNG/JPG image to classify.")
        render_precomputed_demo(project)


def _demo_text_classification(project):
    """Interactive text classification demo."""
    text = st.text_area("Enter text to classify", placeholder="Type a movie review...", height=120,
                        key="demo_txt_cls")
    if st.button("Classify", key="btn_txt_cls"):
        if text.strip():
            st.markdown("**Prediction (demo mode):**")
            # Simple keyword heuristic for demo
            positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'wonderful', 'best', 'fantastic'}
            negative_words = {'bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'poor', 'horrible'}
            words = set(text.lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            if pos > neg:
                st.success("Positive sentiment (demo heuristic)")
            elif neg > pos:
                st.error("Negative sentiment (demo heuristic)")
            else:
                st.warning("Neutral / uncertain (demo heuristic)")
            st.caption("Note: This is a keyword heuristic. The actual model uses BERT-based classification.")
        else:
            st.warning("Please enter some text.")


def _demo_text_generation(project):
    """Interactive text generation demo."""
    prompt = st.text_input("Prompt", value="Once upon a time", key="demo_gen")
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Max tokens", 20, 200, 80, key="demo_gen_len")
    with col2:
        temp = st.slider("Temperature", 0.1, 2.0, 0.8, key="demo_gen_temp")

    if st.button("Generate", key="btn_gen"):
        st.markdown("**Generated text (demo mode):**")
        st.info("Live generation requires a trained GPT-2 model. "
                "See training curves and perplexity metrics above for model quality.")
        render_precomputed_demo(project)


def _demo_recommender(project):
    """Interactive recommender demo."""
    st.markdown("Select a user to get recommendations:")
    user_id = st.number_input("User ID", min_value=1, max_value=943, value=1, key="demo_rec_user")
    n_recs = st.slider("Number of recommendations", 5, 20, 10, key="demo_rec_n")

    if st.button("Get Recommendations", key="btn_rec"):
        st.markdown(f"**Top {n_recs} recommendations for User {user_id} (demo mode):**")
        # Show sample movies
        sample_movies = [
            "Star Wars (1977)", "Fargo (1996)", "Schindler's List (1993)",
            "Shawshank Redemption (1994)", "Pulp Fiction (1994)", "Silence of the Lambs (1991)",
            "Raiders of the Lost Ark (1981)", "Good Will Hunting (1997)",
            "Godfather (1972)", "Toy Story (1995)", "12 Angry Men (1957)",
            "Casablanca (1942)", "Rear Window (1954)", "Amadeus (1984)",
            "Alien (1979)", "Princess Bride (1987)", "Braveheart (1995)",
            "Matrix (1999)", "Fight Club (1999)", "Forrest Gump (1994)",
        ]
        import random
        random.seed(user_id)
        recs = random.sample(sample_movies, min(n_recs, len(sample_movies)))
        for i, movie in enumerate(recs, 1):
            score = round(random.uniform(3.5, 5.0), 2)
            st.markdown(f"{i}. **{movie}** - predicted rating: `{score}`")
        st.caption("Demo mode uses synthetic data. Train the model for real recommendations.")


def _demo_time_series(project):
    """Interactive time series forecasting demo."""
    st.markdown("Adjust forecast parameters:")
    horizon = st.slider("Forecast horizon (steps)", 10, 100, 24, key="demo_ts_horizon")
    lookback = st.slider("Lookback window", 24, 168, 96, key="demo_ts_lookback")

    if st.button("Forecast", key="btn_ts"):
        import plotly.graph_objects as go
        import numpy as np

        np.random.seed(42)
        t = np.arange(lookback + horizon)
        # Generate synthetic time series
        historical = np.sin(t[:lookback] * 2 * np.pi / 24) * 0.5 + np.random.normal(0, 0.1, lookback)
        forecast = np.sin(t[lookback:] * 2 * np.pi / 24) * 0.5 + np.random.normal(0, 0.05, horizon)
        upper = forecast + 0.3
        lower = forecast - 0.3

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(lookback)), y=historical.tolist(),
                                 name='Historical', line=dict(color='#2563eb', width=2)))
        fig.add_trace(go.Scatter(x=list(range(lookback, lookback + horizon)), y=forecast.tolist(),
                                 name='Forecast', line=dict(color='#d97706', width=2)))
        fig.add_trace(go.Scatter(x=list(range(lookback, lookback + horizon)), y=upper.tolist(),
                                 name='Upper bound', line=dict(color='#d97706', width=0),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=list(range(lookback, lookback + horizon)), y=lower.tolist(),
                                 name='Lower bound', line=dict(color='#d97706', width=0),
                                 fill='tonexty', fillcolor='rgba(217,119,6,0.15)',
                                 showlegend=False))
        fig.update_layout(
            title="Time Series Forecast (Demo)",
            font=dict(family="Source Sans 3"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        fig.update_xaxes(title_text="Time Step", gridcolor="#e7e5e4")
        fig.update_yaxes(title_text="Value", gridcolor="#e7e5e4")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.caption("Demo mode uses synthetic data. Train the LSTM model for real forecasts.")


def _demo_anomaly_detection(project):
    """Interactive anomaly detection demo."""
    st.markdown("Generate synthetic data to detect anomalies:")
    n_samples = st.slider("Number of samples", 50, 500, 200, key="demo_ad_n")
    contamination = st.slider("Anomaly ratio", 0.01, 0.2, 0.05, key="demo_ad_cont")

    if st.button("Detect Anomalies", key="btn_ad"):
        import plotly.graph_objects as go
        import numpy as np

        np.random.seed(42)
        n_normal = int(n_samples * (1 - contamination))
        n_anomaly = n_samples - n_normal

        normal = np.random.randn(n_normal, 2) * 0.5
        anomalies = np.random.randn(n_anomaly, 2) * 0.5 + np.array([3, 3])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal[:, 0].tolist(), y=normal[:, 1].tolist(),
                                 mode='markers', name='Normal',
                                 marker=dict(color='#059669', size=6)))
        fig.add_trace(go.Scatter(x=anomalies[:, 0].tolist(), y=anomalies[:, 1].tolist(),
                                 mode='markers', name='Anomaly',
                                 marker=dict(color='#e11d48', size=8, symbol='x')))
        fig.update_layout(
            title="Anomaly Detection (Demo)",
            font=dict(family="Source Sans 3"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
        )
        fig.update_xaxes(title_text="Feature 1", gridcolor="#e7e5e4")
        fig.update_yaxes(title_text="Feature 2", gridcolor="#e7e5e4")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.metric("Detected Anomalies", f"{n_anomaly} / {n_samples}")
        st.caption("Demo mode uses synthetic data. Train the VAE/Autoencoder for real anomaly detection.")
