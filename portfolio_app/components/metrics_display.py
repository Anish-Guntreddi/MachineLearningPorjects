"""
Plotly chart components for metrics visualization.
"""
import plotly.graph_objects as go
from components.model_loader import PLOTLY_COLORS


def plotly_layout(title="", height=400):
    """Standard Plotly layout matching portfolio theme."""
    return go.Layout(
        title=dict(text=title, font=dict(family="JetBrains Mono", size=14, color="#1c1917")),
        font=dict(family="Source Sans 3", size=12, color="#57534e"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(font=dict(size=11)),
    )


def training_curves(history, metric_pairs=None):
    """Plot training/validation loss and accuracy curves."""
    if not history:
        return None

    fig = go.Figure(layout=plotly_layout("Training History", height=350))

    if 'train_loss' in history:
        epochs = list(range(1, len(history['train_loss']) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['train_loss'],
            name='Train Loss', mode='lines',
            line=dict(color=PLOTLY_COLORS[0], width=2),
        ))
    if 'val_loss' in history:
        epochs = list(range(1, len(history['val_loss']) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            name='Val Loss', mode='lines',
            line=dict(color=PLOTLY_COLORS[1], width=2, dash='dash'),
        ))

    fig.update_xaxes(title_text="Epoch", gridcolor="#e7e5e4", gridwidth=1)
    fig.update_yaxes(title_text="Loss", gridcolor="#e7e5e4", gridwidth=1)
    return fig


def accuracy_curves(history):
    """Plot training/validation accuracy curves."""
    if not history:
        return None

    has_acc = any(k for k in history if 'acc' in k.lower())
    if not has_acc:
        return None

    fig = go.Figure(layout=plotly_layout("Accuracy", height=350))

    for i, (key, values) in enumerate(history.items()):
        if 'acc' in key.lower():
            epochs = list(range(1, len(values) + 1))
            label = key.replace('_', ' ').title()
            dash = 'dash' if 'val' in key else 'solid'
            fig.add_trace(go.Scatter(
                x=epochs, y=values,
                name=label, mode='lines',
                line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2, dash=dash),
            ))

    fig.update_xaxes(title_text="Epoch", gridcolor="#e7e5e4")
    fig.update_yaxes(title_text="Accuracy", gridcolor="#e7e5e4")
    return fig


def model_comparison_chart(comparisons):
    """Bar chart comparing different model architectures."""
    if not comparisons:
        return None

    fig = go.Figure(layout=plotly_layout("Model Comparison", height=300))

    names = [c.get('model', '') for c in comparisons]
    values = []
    metric_name = ""
    for c in comparisons:
        for k, v in c.items():
            if k not in ('model', 'params') and isinstance(v, (int, float)):
                values.append(v)
                metric_name = k.replace('_', ' ').title()
                break

    colors = [PLOTLY_COLORS[i % len(PLOTLY_COLORS)] for i in range(len(names))]

    fig.add_trace(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition='outside',
        textfont=dict(family="JetBrains Mono", size=11),
    ))

    fig.update_xaxes(gridcolor="#e7e5e4")
    fig.update_yaxes(title_text=metric_name, gridcolor="#e7e5e4")
    fig.update_layout(showlegend=False)
    return fig


def per_class_heatmap(per_class_metrics):
    """Heatmap of per-class precision, recall, F1."""
    if not per_class_metrics:
        return None

    classes = [c.get('class', f'Class {i}') for i, c in enumerate(per_class_metrics)]
    metrics = ['precision', 'recall', 'f1']
    z = []
    for m in metrics:
        row = [c.get(m, 0) for c in per_class_metrics]
        z.append(row)

    fig = go.Figure(layout=plotly_layout("Per-Class Metrics", height=max(250, len(classes) * 20 + 100)))

    fig.add_trace(go.Heatmap(
        z=z,
        x=classes,
        y=[m.title() for m in metrics],
        colorscale=[[0, '#f0fdfa'], [0.5, '#0d9488'], [1, '#0f766e']],
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=10),
        showscale=False,
    ))

    return fig


def radar_chart(projects_data):
    """Radar chart comparing multiple projects on normalized metrics."""
    if not projects_data:
        return None

    fig = go.Figure(layout=plotly_layout("Project Comparison", height=450))

    for i, (name, metrics) in enumerate(projects_data.items()):
        categories = list(metrics.keys())
        values = list(metrics.values())
        values.append(values[0])  # close the polygon
        categories.append(categories[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name,
            line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)]),
            fillcolor=f"rgba({int(PLOTLY_COLORS[i % len(PLOTLY_COLORS)][1:3], 16)}, "
                      f"{int(PLOTLY_COLORS[i % len(PLOTLY_COLORS)][3:5], 16)}, "
                      f"{int(PLOTLY_COLORS[i % len(PLOTLY_COLORS)][5:7], 16)}, 0.1)",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e7e5e4"),
            angularaxis=dict(gridcolor="#e7e5e4"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig
