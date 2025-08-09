import os
from typing import Optional
from .utils.tools import tool

@tool()
def write_file(path: str, content: str, wait_for: Optional[str] = None):
    """
    Create or overwrite a file with the given content.

    Args:
        path: The path where the file should be created or overwritten
        content: The content to write to the file
        wait_for: Reference outputs from prior actions.
    """
    try:       
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return (False, f"File written at {path}")
    except Exception as e:
        return (True, f"Failed to write file at {path}: {str(e)}")
