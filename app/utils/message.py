import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

class BaseMessage:
    def __init__(
            self,
            content: str,
            role: str = "user", 
            metadata: Optional[Dict[str, Any]] = None
        ):
        self.role = role
        self.content = content
        self.metadata = metadata
        self.timestamp = datetime.datetime.now().isoformat()
        self.id = str(uuid4())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content='{self.content}', timestamp='{self.timestamp}')"

class UserMessage(BaseMessage):
    def __init__(
            self, 
            content: str,
    ):
        super().__init__(
            content=content,
            role="user"
        )

class SystemMessage(BaseMessage):
    def __init__(
            self, 
            content: str,
    ):
        super().__init__(
            content=content,
            role="system",
        )

class AIMessage(BaseMessage):
    def __init__(
            self, 
            content: str,
            tool_calls: List[Dict[str, Any]] = None, 
            metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            content=content,
            role="assistant",
            metadata=metadata
        )
        self.tool_calls = tool_calls

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content='{self.content}', tool_calls='{self.tool_calls}')"

class ToolMessage(BaseMessage):
    def __init__(
            self,
            name: str, 
            content: str,
            tool_call_id: str, 
            metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            content=content,
            role="tool",
            metadata=metadata
        )
        self.name = name
        self.tool_call_id = tool_call_id

class FunctionMessage(BaseMessage):
    def __init__(
            self,
            name: str, 
            content: str,
            tool_call_id: str, 
            metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            content=content,
            role="function",
            metadata=metadata
        )
        self.name = name
        self.tool_call_id = tool_call_id
