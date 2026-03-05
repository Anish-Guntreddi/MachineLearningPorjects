"""
Vision-specific notebook cells for projects 01, 02, 03
"""
import nbformat


def get_extra_cells(config):
    """Return extra cells for vision projects"""
    cells = []

    cells.append(nbformat.v4.new_markdown_cell("### Domain-Specific: Vision Analysis"))

    cells.append(nbformat.v4.new_code_cell("""# Visualize model predictions with confidence
model.eval()

test_batch = next(iter(test_loader))
if isinstance(test_batch, (list, tuple)):
    images, targets = test_batch
else:
    images, targets = test_batch, None

if isinstance(images, list):
    sample = images[:4]
    with torch.no_grad():
        preds = model([img.to(device) for img in sample])

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]
    for i, (img, pred) in enumerate(zip(sample, preds)):
        axes[i].imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
        if 'scores' in pred:
            n_det = (pred['scores'] > 0.5).sum().item()
            axes[i].set_title(f'{n_det} detections')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
else:
    sample = images[:8].to(device)
    with torch.no_grad():
        outputs = model(sample)
        if isinstance(outputs, torch.Tensor):
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < len(sample):
            img = sample[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            if isinstance(outputs, torch.Tensor):
                ax.set_title(f'Pred: {preds[i].item()} ({confs[i].item():.2f})')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print("Vision analysis complete.")
"""))

    return cells
