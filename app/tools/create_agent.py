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
def create_agent(
    name: str, 
    system_prompt: str
):
    """
    Create an AI agent

    Args:
        name: The name of the AI agent
        system_prompt: Detailed system prompt to guide AI behaviour
    """
    agent = ReAct(
        system_prompt=system_prompt,
    )

    Register().set(name, agent)

    return (False, f"Agent {name} created successfully")
    