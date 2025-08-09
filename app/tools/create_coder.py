from typing import List
from .utils.tools import tool

from app.agent import (
    ReAct,
    Register
)

from .read_file import read_file
from .append_file import append_file
from .write_file import write_file

@tool()
def create_coder(
    name: str, 
    system_prompt: str
):
    """
    Create a specialized AI agent for coding

    Args:
        name: The name of the AI agent
        system_prompt: Detailed system prompt to guide AI behaviour
    """
    agent = ReAct(
        coder=True,
        system_prompt=f"""
{system_prompt}

You are a coding AI agent. Your primary function is to generate code based on user requests.

Your response must be *only* the code itself. Do not include any of the following:
* Markdown formatting (e.g., ```python)
* Explanations or any other natural language text *outside* of the code
* Installation instructions (e.g., pip install)

---

**Additional Directives:**

1.  **Code Comments:** You are permitted—and encouraged—to use comments *within* the code itself to provide explanations, clarify logic, and document key sections. These comments should be concise and helpful.
2.  **Clarity and Conciseness:** The code should be clean, efficient, and directly solve the user's request. Avoid unnecessary complexity.
3.  **Dependencies:** If the code requires external libraries, assume they are already installed. Import all necessary libraries at the top of the file.
4.  **Error Handling:** Include basic, appropriate error handling where it makes sense (e.g., handling file not found errors, invalid user input).
""",
    )

    Register().set(name, agent)

    return (False, f"Agent {name} created successfully")
    