from .utils.tools import tool

@tool()
def join(defer_to_replanner: str):
    """
    Synthesizes results from all previous actions. Required as the final step.

    Args:
        defer_to_replanner: A comprehensive 
    """
    print(defer_to_replanner)

    return (False, defer_to_replanner)