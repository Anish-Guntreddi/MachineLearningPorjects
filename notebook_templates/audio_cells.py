"""
Audio-specific notebook cells for projects 07, 08
"""
import nbformat


def get_extra_cells(config):
    """Return extra cells for audio projects"""
    cells = []

    cells.append(nbformat.v4.new_markdown_cell("### Domain-Specific: Audio Analysis"))

    cells.append(nbformat.v4.new_code_cell("""# Audio-specific analysis
print("Audio feature analysis:")

test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    for key, val in test_batch.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: shape={val.shape}")
elif isinstance(test_batch, (list, tuple)):
    for i, item in enumerate(test_batch):
        if isinstance(item, torch.Tensor):
            print(f"  Tensor {i}: shape={item.shape}")

# Visualize spectrograms if applicable
try:
    features = test_batch[0] if isinstance(test_batch, (list, tuple)) else test_batch.get('features', test_batch.get('input_values'))
    if features is not None and features.dim() >= 2:
        fig, axes = plt.subplots(1, min(4, features.shape[0]), figsize=(16, 4))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for i, ax in enumerate(axes):
            if features.dim() == 3:
                ax.imshow(features[i].squeeze().numpy(), aspect='auto', origin='lower')
            elif features.dim() == 2:
                ax.plot(features[i].numpy())
            ax.set_title(f'Sample {i}')
        plt.suptitle('Audio Features')
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Could not visualize audio features: {e}")

print("Audio analysis complete.")
"""))

    return cells
