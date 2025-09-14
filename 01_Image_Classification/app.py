"""
Gradio demo app for image classification
"""
import gradio as gr
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Tuple
import os

from inference import ImageClassifier


# Initialize classifier
MODEL_PATH = "checkpoints/best_model.pth"
MODEL_NAME = "resnet18"

# Check if model exists, otherwise use untrained model for demo
if not os.path.exists(MODEL_PATH):
    print("Warning: No trained model found. Using untrained model for demo.")
    os.makedirs("checkpoints", exist_ok=True)

classifier = ImageClassifier(
    model_path=MODEL_PATH,
    model_name=MODEL_NAME,
    device='auto'
)


def classify_image(image: Image.Image) -> Dict:
    """
    Classify uploaded image
    
    Args:
        image: PIL Image
    
    Returns:
        Dictionary of class probabilities
    """
    # Save temporary image
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Get predictions
    results = classifier.predict(temp_path, top_k=10)
    
    # Format for Gradio output
    predictions = {}
    for pred in results['predictions']:
        predictions[pred['class']] = float(pred['confidence'])
    
    # Clean up
    os.remove(temp_path)
    
    return predictions


def classify_with_tta(image: Image.Image, num_augmentations: int = 10) -> Tuple[Dict, str]:
    """
    Classify with test-time augmentation
    
    Args:
        image: PIL Image
        num_augmentations: Number of augmentations
    
    Returns:
        Tuple of predictions and info string
    """
    # Save temporary image
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    
    # Get predictions with TTA
    results = classifier.test_time_augmentation(temp_path, num_augmentations)
    
    # Format predictions
    predictions = {}
    for pred in results['predictions']:
        predictions[pred['class']] = float(pred['confidence'])
    
    # Create info string
    info = f"Test-Time Augmentation with {num_augmentations} augmentations\n"
    info += f"Top prediction: {results['predictions'][0]['class']} "
    info += f"({results['predictions'][0]['confidence']:.2%} confidence)"
    
    # Clean up
    os.remove(temp_path)
    
    return predictions, info


def create_confidence_plot(predictions: Dict) -> Image.Image:
    """
    Create a bar plot of prediction confidences
    
    Args:
        predictions: Dictionary of class probabilities
    
    Returns:
        PIL Image of the plot
    """
    # Sort predictions by confidence
    sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(sorted_preds.keys())
    confidences = list(sorted_preds.values())
    
    bars = ax.barh(classes, confidences)
    
    # Color bars based on confidence
    colors = ['green' if c > 0.5 else 'orange' if c > 0.3 else 'red' for c in confidences]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Confidence')
    ax.set_title('Top 5 Predictions')
    ax.set_xlim([0, 1])
    
    # Add percentage labels
    for i, (c, conf) in enumerate(zip(classes, confidences)):
        ax.text(conf + 0.01, i, f'{conf:.1%}', va='center')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


# Create Gradio interface
with gr.Blocks(title="Image Classification Demo") as demo:
    gr.Markdown("""
    # 🖼️ Image Classification Demo
    
    Upload an image to classify it into one of 10 categories:
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """)
    
    with gr.Tab("Standard Classification"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                classify_btn = gr.Button("Classify", variant="primary")
                
            with gr.Column():
                output_label = gr.Label(num_top_classes=5, label="Predictions")
                confidence_plot = gr.Image(label="Confidence Visualization")
        
        def classify_and_plot(image):
            if image is None:
                return None, None
            predictions = classify_image(image)
            plot = create_confidence_plot(predictions)
            return predictions, plot
        
        classify_btn.click(
            fn=classify_and_plot,
            inputs=input_image,
            outputs=[output_label, confidence_plot]
        )
    
    with gr.Tab("Test-Time Augmentation"):
        with gr.Row():
            with gr.Column():
                tta_image = gr.Image(type="pil", label="Upload Image")
                num_aug_slider = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of Augmentations"
                )
                tta_btn = gr.Button("Classify with TTA", variant="primary")
                
            with gr.Column():
                tta_output = gr.Label(num_top_classes=5, label="Averaged Predictions")
                tta_info = gr.Textbox(label="TTA Information", lines=3)
        
        tta_btn.click(
            fn=classify_with_tta,
            inputs=[tta_image, num_aug_slider],
            outputs=[tta_output, tta_info]
        )
    
    with gr.Tab("Model Information"):
        gr.Markdown(f"""
        ### Model Details
        - **Architecture**: {MODEL_NAME}
        - **Number of Classes**: 10 (CIFAR-10)
        - **Input Size**: 32x32 pixels
        - **Device**: {classifier.device}
        
        ### Performance Metrics
        - Training Accuracy: ~95% (with proper training)
        - Inference Time: ~5-10ms per image
        
        ### Tips for Best Results
        1. Use clear, well-lit images
        2. Center the object in the frame
        3. Avoid cluttered backgrounds
        4. Try Test-Time Augmentation for more robust predictions
        """)
    
    # Add examples
    gr.Examples(
        examples=[
            ["examples/airplane.jpg"],
            ["examples/car.jpg"],
            ["examples/bird.jpg"],
            ["examples/cat.jpg"],
            ["examples/dog.jpg"]
        ],
        inputs=input_image,
        outputs=[output_label, confidence_plot],
        fn=classify_and_plot,
        cache_examples=False
    )


if __name__ == "__main__":
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True to create a public link
    )