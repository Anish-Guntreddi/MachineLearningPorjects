"""
Multimodal-specific notebook cells for project 12
"""
import nbformat


def get_extra_cells(config):
    """Return extra cells for multimodal projects"""
    cells = []

    cells.append(nbformat.v4.new_markdown_cell("### Domain-Specific: Multimodal Fusion Analysis"))

    cells.append(nbformat.v4.new_code_cell("""# Multimodal analysis
print("Multimodal data analysis:")

# Check modality inputs
test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    for key, val in test_batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}")

# Ablation study: test each modality separately
print("\\nAblation study:")
model.eval()

try:
    if hasattr(model, 'forward'):
        # Test with full input
        with torch.no_grad():
            if isinstance(test_batch, dict):
                sample = {k: v[:4].to(device) for k, v in test_batch.items()
                         if isinstance(v, torch.Tensor)}
                full_output = model(**sample)
                print(f"  Full model output shape: {full_output.shape if isinstance(full_output, torch.Tensor) else type(full_output)}")
except Exception as e:
    print(f"  Ablation study error: {e}")

print("\\nMultimodal analysis complete.")
"""))

    return cells
