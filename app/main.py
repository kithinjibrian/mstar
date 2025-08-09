from app.tools import (
    write_file,
    read_file,
    append_file,
    search_internet,
    create_agent,
    create_coder,
    instruct_agent,
    bash,
    python,
    close_python,
)

from app.agent import (
    Agent,
    ReAct
)

def main():
    agent = ReAct(
        tools=[
            write_file,
            read_file,
            append_file,
            search_internet,
            create_agent,
            create_coder,
            instruct_agent,
            bash,
            python,
            close_python,
        ],
    )
    agent.run("Create a new expressjs server project")

# What led to Kenya's 2025 GenZ protest?