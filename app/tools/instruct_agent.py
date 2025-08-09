import re
import time
from .utils.tools import tool
from app.agent import (
    ReAct,
    Register
)

def extract_code(text):
    """
    Return the first fenced code block if present.
    Otherwise, return the full text as-is (assuming it's all code).
    """
    pattern = r"```(?:\w*\n)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()



@tool()
def instruct_agent(agent_name: str, user_prompt: str):
    """
    Instruct an AI agent with retry mechanism

    Args:
        agent_name: The name of the AI agent
        user_prompt: Prompt an AI to perform a task
    """
    for attempt in range(3):
        agent = Register().get(agent_name)
        
        if agent:
            if isinstance(agent, ReAct):
                response = agent.run(user_prompt)
                if agent.coder:
                    code = extract_code(response.content)
                    return (False, code)
                
                return (False, response.content)
            else:
                return (True, f"Agent {agent_name} is not of type ReAct.")
        
        time.sleep(1)

    return (True, f"Agent {agent_name} not found after {3} attempts.")
