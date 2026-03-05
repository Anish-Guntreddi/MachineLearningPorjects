"""
Tabular-specific notebook cells for projects 09, 10, 11
"""
import nbformat


def get_extra_cells(config):
    """Return extra cells for tabular projects"""
    cells = []

    cells.append(nbformat.v4.new_markdown_cell("### Domain-Specific: Tabular Data Analysis"))

    cells.append(nbformat.v4.new_code_cell("""# Tabular data analysis
print("Data statistics:")

test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    for key, val in test_batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}, mean={val.float().mean():.4f}, std={val.float().std():.4f}")
elif isinstance(test_batch, (list, tuple)):
    for i, item in enumerate(test_batch):
        if isinstance(item, torch.Tensor):
            print(f"  Tensor {i}: shape={item.shape}, mean={item.float().mean():.4f}, std={item.float().std():.4f}")

# Visualize predictions vs actual if applicable
try:
    model.eval()
    with torch.no_grad():
        if isinstance(test_batch, dict):
            inputs = test_batch.get('input', test_batch.get('user_ids'))
            if inputs is not None:
                sample_input = {k: v[:8].to(device) for k, v in test_batch.items()
                               if isinstance(v, torch.Tensor) and k not in ('target', 'label', 'labels', 'rating')}
                outputs = model(**sample_input)
                print(f"\\nSample output shape: {outputs.shape if isinstance(outputs, torch.Tensor) else type(outputs)}")
except Exception as e:
    print(f"Could not generate predictions: {e}")

print("\\nTabular analysis complete.")
"""))

    return cells
