import ast
from dataclasses import dataclass
import re
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    TypedDict,
    Union
)

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = "<END_OF_PLAN>"

class Task(TypedDict):
    idx: int
    name: str
    tool: Any
    args: Dict
    dependencies: Dict[str, list]
    thought: Optional[str]

@dataclass
class A:
    __args__: Dict

def _ast_parse(arg: str) -> Any:
    try:
        return ast.literal_eval(arg)
    except:  # noqa
        return arg

def _parse_args(args: str, tool: Any) -> list[Any]:
    """Parse arguments from a string."""
    if args == "":
        return ()
    if isinstance(tool, str):
        return ()
    extracted_args = {}
    tool_key = None
    prev_idx = None
    for key in tool.__args__:
        # Split if present
        if f"{key}=" in args:
            idx = args.index(f"{key}=")
            if prev_idx is not None:
                extracted_args[tool_key] = _ast_parse(
                    args[prev_idx:idx].strip().rstrip(",")
                )
            args = args.split(f"{key}=", 1)[1]
            tool_key = key
            prev_idx = 0
    if prev_idx is not None:
        extracted_args[tool_key] = _ast_parse(
            args[prev_idx:].strip().rstrip(",").rstrip(")")
        )
    return extracted_args

def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers

def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Dict[str, Any]
) -> dict[str, list[str]]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        return list(range(1, idx))
    return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]

def instantiate_task(
    tools: List[Any],
    idx: int,
    tool_name: str,
    args: Union[str, Any],
    thought: Optional[str] = None,
) -> Task:
    if tool_name == "join":
        tool = "join"
    else:
        try:
            tool = tools[[tool.__name__ for tool in tools].index(tool_name)]
        except ValueError as e:
            return None
    
    if tool_name == "join":            
        tool_args = _parse_args(args, A(__args__= {
                "defer_to_replanner": ""
            }
        ))
    else:
        tool_args = _parse_args(args, tool)

    dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)

    return Task(
        idx=idx,
        name=tool_name,
        tool=tool,
        args=tool_args,
        dependencies=dependencies,
        thought=thought,
    )

class Parser:
    def __init__(self, tools):
        self.tools = tools
        self.reset_stream()

    def reset_stream(self):
        """Reset the streaming state to start fresh."""
        self._buffer = []
        self._current_thought = None
        self._completed = False

    def stream(self, chunk: str) -> Iterator[Task]:
        """
        Process a single chunk of streaming text and yield any completed tasks.
        
        Args:
            chunk: String chunk to process (can be any size)
            
        Yields:
            Task: Completed tasks as they are parsed
        """
        if self._completed:
            return
            
        # Check for end of plan
        if END_OF_PLAN in chunk:
            self._completed = True
            # Process any remaining content before the END_OF_PLAN marker
            end_idx = chunk.index(END_OF_PLAN)
            chunk = chunk[:end_idx]
            
        if not chunk:
            return
            
        self._buffer.append(chunk)
        
        # # Only process when we have newlines
        if "\n" in chunk:
            buffer_text = "".join(self._buffer)
            lines = buffer_text.split("\n")
            
            # Keep the last line in buffer (might be incomplete)
            incomplete_line = lines[-1]
            complete_lines = lines[:-1]
            
            # Process all complete lines
            for line in complete_lines:
                task, self._current_thought = self._parse_task(line, self._current_thought)
                if task:
                    yield task
            
            # Reset buffer with incomplete line
            self._buffer = [incomplete_line] if incomplete_line else []

    def finalize_stream(self) -> Iterator[Task]:
        """
        Process any remaining buffered content and yield final tasks.
        Call this when streaming is complete.
        
        Yields:
            Task: Any remaining tasks in the buffer
        """
        if self._buffer:
            remaining_text = "".join(self._buffer)
            if remaining_text.strip():
                task, _ = self._parse_task(remaining_text, self._current_thought)
                if task:
                    yield task
        self.reset_stream()

    def is_complete(self) -> bool:
        """Check if the END_OF_PLAN marker was encountered."""
        return self._completed

    def get_buffer_content(self) -> str:
        """Get current buffered content (for debugging)."""
        return "".join(self._buffer)

    def _parse_task(self, line: str, thought: Optional[str] = None):
        """Parse a single line into a task."""
        task = None
        
        if match := re.match(THOUGHT_PATTERN, line):
            thought = match.group(1)
        elif match := re.match(ACTION_PATTERN, line):
            idx, tool_name, args, _ = match.groups()
            idx = int(idx)
            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )
            thought = None 

        return task, thought

    # Legacy methods for backward compatibility
    def _transform(self, input: Iterator[Union[str]]) -> Iterator[Task]:
        texts = []
        thought = None

        for chunk in input:
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self.ingest_token(text, texts, thought):
                yield task
        if texts:
            task, _ = self._parse_task("".join(texts), thought)
            if task:
                yield task

    def parse(self, text: str):
        """Parse complete text (batch mode)."""
        return list(self._transform([text]))

    def ingest_token(self, token: str, buffer: List[str], thought: Optional[str]):
        """Legacy streaming method (use stream_chunk instead)."""
        buffer.append(token)
        if "\n" in token:
            buffer_ = "".join(buffer).split("\n")
            suffix = buffer_[-1]
            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought
            buffer.clear()
            buffer.append(suffix)