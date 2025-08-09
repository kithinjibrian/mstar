import os
from typing import Optional
from .utils.tools import tool

@tool()
def read_file(path: str, wait_for: Optional[str] = None):
    """
    Read and return the content of a file.

    Args:
        path: The path to the file to read
        wait_for: Reference outputs from other actions.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return (False, content)
    except FileNotFoundError:
        return (True, f"File not found at {path}")
    except Exception as e:
        return (True, f"Failed to read file at {path}: {str(e)}")