"""
Notebook generator for ML Portfolio.
Generates one Jupyter notebook per project from shared templates and per-project YAML configs.

Usage:
    python generate_notebooks.py              # Generate all 12
    python generate_notebooks.py --project 01 # Generate single
    python generate_notebooks.py --dry-run    # Validate configs only
"""
import os
import sys
import argparse
import yaml
import nbformat
from pathlib import Path

# Add templates to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from notebook_templates import common_cells
from notebook_templates import vision_cells, nlp_cells, audio_cells, tabular_cells, multimodal_cells

DOMAIN_MODULES = {
    'vision': vision_cells,
    'nlp': nlp_cells,
    'audio': audio_cells,
    'tabular': tabular_cells,
    'multimodal': multimodal_cells,
}


def load_config(config_path: str) -> dict:
    """Load and validate a project YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_keys = ['project_num', 'project_name', 'project_dir', 'domain',
                     'description', 'model', 'data', 'training', 'evaluation']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in {config_path}")

    return config


def generate_notebook(config: dict) -> nbformat.NotebookNode:
    """Generate a notebook from a project config."""
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
    nb.metadata.language_info = {
        'name': 'python',
        'version': '3.10.0'
    }

    cells = []

    # Cell 1: Title
    cells.append(common_cells.title_cell(config))

    # Cell 2: Setup and device detection
    cells.append(common_cells.setup_cell(config))

    # Cell 3: Configuration
    cells.append(common_cells.config_cell(config))

    # Cell 4: Data loading
    if config['data'].get('has_data_loader', True):
        cells.append(common_cells.data_loading_cell(config))
    else:
        inline = config['data'].get('inline_data', '# No data loader available')
        cells.append(common_cells.inline_data_cell(inline))

    # Cell 5: Preprocessing
    cells.append(common_cells.preprocessing_cell(config))

    # Cell 6: Model architecture
    cells.append(common_cells.model_cell(config))

    # Cell 7: Training
    if config['training'].get('has_train_py', True):
        cells.append(common_cells.training_cell(config))
    else:
        inline = config['training'].get('inline_training', '# No train.py available')
        cells.append(common_cells.inline_training_cell(inline))

    # Cell 8: Training visualization
    cells.append(common_cells.visualization_cell(config))

    # Cell 9: Evaluation
    cells.append(common_cells.evaluation_cell(config))

    # Cell 10: Predictions
    cells.append(common_cells.predictions_cell(config))

    # Domain-specific extra cells (between 10 and 11)
    domain_mod = DOMAIN_MODULES.get(config['domain'])
    if domain_mod:
        cells.extend(domain_mod.get_extra_cells(config))

    # Cell 11: Export results
    cells.append(common_cells.export_cell(config))

    # Cell 12: Summary
    cells.append(common_cells.summary_cell(config))

    nb.cells = cells
    return nb


def main():
    parser = argparse.ArgumentParser(description='Generate Jupyter notebooks for ML projects')
    parser.add_argument('--project', type=str, default=None,
                        help='Generate for specific project (e.g., "01" or "01_config")')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configs without generating notebooks')
    parser.add_argument('--configs-dir', type=str, default='notebook_configs',
                        help='Directory containing YAML configs')
    args = parser.parse_args()

    configs_dir = ROOT / args.configs_dir

    if not configs_dir.exists():
        print(f"Error: Configs directory not found: {configs_dir}")
        sys.exit(1)

    # Find config files
    if args.project:
        project_id = args.project.replace('_config', '').replace('.yaml', '')
        config_files = list(configs_dir.glob(f'{project_id}*_config.yaml'))
        if not config_files:
            config_files = list(configs_dir.glob(f'*{project_id}*.yaml'))
        if not config_files:
            print(f"Error: No config found for project '{args.project}'")
            sys.exit(1)
    else:
        config_files = sorted(configs_dir.glob('*_config.yaml'))

    if not config_files:
        print("Error: No config files found")
        sys.exit(1)

    print(f"Found {len(config_files)} config file(s)")

    # Process each config
    success = 0
    errors = 0

    for config_path in config_files:
        try:
            config = load_config(config_path)
            project_dir = config['project_dir']
            project_name = config['project_name']

            if args.dry_run:
                print(f"  [OK] {config_path.name}: {project_name} ({config['domain']})")
                success += 1
                continue

            # Generate notebook
            nb = generate_notebook(config)

            # Write notebook
            output_dir = ROOT / project_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'notebook.ipynb'

            with open(output_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            cell_count = len(nb.cells)
            print(f"  [OK] {project_name}: {output_path} ({cell_count} cells)")
            success += 1

        except Exception as e:
            print(f"  [ERROR] {config_path.name}: {e}")
            errors += 1

    print(f"\nDone: {success} succeeded, {errors} failed")

    if errors > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
