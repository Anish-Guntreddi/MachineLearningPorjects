"""
Notebook renderer component for displaying Jupyter notebook code and outputs inline.
"""
import json
import base64
import html
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st


@st.cache_data
def load_notebook(notebook_path: str) -> Optional[Dict]:
    """Load and parse a Jupyter notebook file."""
    path = Path(notebook_path)
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _render_markdown_cell(cell: Dict):
    """Render a markdown cell."""
    source = ''.join(cell.get('source', []))
    if source.strip():
        st.markdown(source)


def _get_output_text(output: Dict) -> Optional[str]:
    """Extract text content from a cell output."""
    if output.get('output_type') == 'stream':
        return ''.join(output.get('text', []))
    if output.get('output_type') in ('execute_result', 'display_data'):
        data = output.get('data', {})
        if 'text/plain' in data:
            text = data['text/plain']
            return ''.join(text) if isinstance(text, list) else text
    if output.get('output_type') == 'error':
        tb = output.get('traceback', [])
        # Strip ANSI escape codes for clean display
        import re
        ansi_re = re.compile(r'\x1b\[[0-9;]*m')
        return '\n'.join(ansi_re.sub('', line) for line in tb)
    return None


def _get_output_image(output: Dict) -> Optional[bytes]:
    """Extract image data from a cell output."""
    if output.get('output_type') in ('display_data', 'execute_result'):
        data = output.get('data', {})
        if 'image/png' in data:
            img_data = data['image/png']
            if isinstance(img_data, list):
                img_data = ''.join(img_data)
            return base64.b64decode(img_data)
    return None


def _render_code_cell(cell: Dict, show_outputs: bool = True):
    """Render a code cell with its source and outputs."""
    source = ''.join(cell.get('source', []))
    if not source.strip():
        return

    st.code(source, language='python')

    if not show_outputs:
        return

    outputs = cell.get('outputs', [])
    if not outputs:
        return

    for output in outputs:
        # Try image first
        img_data = _get_output_image(output)
        if img_data:
            st.image(img_data)
            continue

        # Try text
        text = _get_output_text(output)
        if text and text.strip():
            # Truncate very long outputs
            lines = text.split('\n')
            if len(lines) > 50:
                truncated = '\n'.join(lines[:50])
                with st.expander(f"Output ({len(lines)} lines — click to expand)"):
                    st.code(truncated + f'\n\n... ({len(lines) - 50} more lines)', language='text')
            else:
                st.code(text, language='text')


def render_notebook(project_path: str, show_outputs: bool = True, cell_filter: str = "All"):
    """
    Render a Jupyter notebook inline in Streamlit.

    Args:
        project_path: Path to the project directory containing notebook.ipynb
        show_outputs: Whether to show cell outputs
        cell_filter: "All", "Code only", or "Markdown only"
    """
    notebook_path = Path(project_path) / 'notebook.ipynb'
    nb = load_notebook(str(notebook_path))

    if nb is None:
        st.info("No notebook found for this project.")
        return

    cells = nb.get('cells', [])
    if not cells:
        st.info("Notebook is empty.")
        return

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        cell_filter = st.selectbox(
            "Show",
            ["All cells", "Code only", "Markdown only"],
            key=f"nb_filter_{project_path}"
        )
    with col2:
        show_outputs = st.checkbox("Show outputs", value=True, key=f"nb_outputs_{project_path}")

    st.divider()

    # Render cells
    code_idx = 0
    for cell in cells:
        cell_type = cell.get('cell_type', '')

        if cell_filter == "Code only" and cell_type != 'code':
            continue
        if cell_filter == "Markdown only" and cell_type != 'markdown':
            continue

        if cell_type == 'markdown':
            _render_markdown_cell(cell)
        elif cell_type == 'code':
            code_idx += 1
            _render_code_cell(cell, show_outputs=show_outputs)
