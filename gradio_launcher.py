"""
What is needed                                                                                                              
                                                                                                                              
gradio_launcher.py — Root-level entry point for PaperAlchemy's Gradio UI.

What it does:
    Adds the project root to sys.path so src.gradio_app imports resolve
    correctly, then delegates to gradio_app.main() to build and launch
    the Gradio interface.

Why it is needed:
    src/gradio_app.py lives inside the src/ package. Running it directly
    (python src/gradio_app.py) can break imports. This launcher sits at
    the project root — the same level as compose.yml and pyproject.toml —
    ensuring all src.* imports resolve cleanly from one known location.

How to use:
    python gradio_launcher.py

    The UI will be available at http://localhost:7861
    Requires the PaperAlchemy API to be running at http://localhost:8000

"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.gradio_app import main

if __name__ == "__main__":
    main()