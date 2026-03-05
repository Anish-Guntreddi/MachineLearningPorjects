"""
Input widgets for interactive demos by project type.
"""
import streamlit as st


def image_upload_demo():
    """Image upload widget for vision projects."""
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="demo_image")
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)
    return uploaded


def text_input_demo(placeholder="Enter text here..."):
    """Text input widget for NLP projects."""
    text = st.text_area("Input Text", placeholder=placeholder, height=150, key="demo_text")
    return text


def text_generation_demo(placeholder="Once upon a time"):
    """Text generation input with parameters."""
    text = st.text_input("Prompt", value=placeholder, key="demo_gen_prompt")
    col1, col2 = st.columns(2)
    with col1:
        max_len = st.slider("Max Length", 20, 200, 100, key="demo_gen_len")
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, key="demo_gen_temp")
    return text, max_len, temperature


def tabular_input_demo(options=None):
    """Selection-based input for tabular projects."""
    if options:
        selection = st.selectbox("Select input", options, key="demo_tabular_select")
        return selection
    return None


def slider_input_demo(label="Input Value", min_val=0.0, max_val=1.0, default=0.5):
    """Slider input for numeric parameters."""
    return st.slider(label, min_val, max_val, default, key=f"demo_slider_{label}")


def render_precomputed_demo(project):
    """Display precomputed demo results from results.yaml."""
    results = project.get('results', {})
    samples = results.get('sample_predictions', [])

    if not samples:
        st.info("No interactive demo available for this project. See metrics and training curves above.")
        return

    st.markdown("#### Sample Predictions")

    for i, sample in enumerate(samples):
        with st.expander(f"Sample {i + 1}: {sample.get('input_description', 'Sample')}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Input:** {sample.get('input_description', 'N/A')}")
                st.markdown(f"**True Label:** `{sample.get('true_label', 'N/A')}`")
            with col2:
                st.markdown(f"**Predicted:** `{sample.get('predicted_label', 'N/A')}`")
                conf = sample.get('confidence', None)
                if conf is not None:
                    st.progress(float(conf), text=f"Confidence: {float(conf):.1%}")
