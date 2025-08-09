import os
from typing import Optional
from .utils.tools import tool

@tool()
def append_file(path: str, content: str, wait_for: Optional[str] = None):
    """
    Append content to an existing file.

    Args:
        path: The path to the file
        content: The content to append to the file
        wait_for: Reference outputs from other actions.
    """
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)
        return (False, f"Content appended to {path}")
    except Exception as e:
        return (True, f"Failed to append to file at {path}: {str(e)}")