"""
NLP-specific notebook cells for projects 04, 05, 06
"""
import nbformat


def get_extra_cells(config):
    """Return extra cells for NLP projects"""
    cells = []

    cells.append(nbformat.v4.new_markdown_cell("### Domain-Specific: NLP Analysis"))

    cells.append(nbformat.v4.new_code_cell("""# NLP-specific analysis
model.eval()

# Show model vocabulary info if available
if hasattr(model, 'config'):
    print(f"Model config vocab size: {getattr(model.config, 'vocab_size', 'N/A')}")
    print(f"Model config hidden size: {getattr(model.config, 'hidden_size', 'N/A')}")

# Sample generation or prediction
test_batch = next(iter(test_loader))
if isinstance(test_batch, dict):
    input_ids = test_batch.get('input_ids', test_batch.get('src_input_ids'))
    if input_ids is not None:
        print(f"\\nSample input shape: {input_ids.shape}")
        print(f"Input token range: [{input_ids.min().item()}, {input_ids.max().item()}]")

print("\\nNLP analysis complete.")
"""))

    return cells
