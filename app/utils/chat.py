from typing import List, Tuple, Dict, Any, Union, Optional
import copy
import re

from .message import (
    SystemMessage,
    UserMessage,
    AIMessage,
    ToolMessage, 
    BaseMessage,
    FunctionMessage
)

class Placeholder:
    """Represents a placeholder for message insertion in templates."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Placeholder('{self.name}')"
    
    def __eq__(self, other):
        return isinstance(other, Placeholder) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

Message = Union[Tuple[str, str], Placeholder, BaseMessage] 

class ChatTemplate:
    """
    A flexible template system for managing chat messages with placeholder substitution.
    
    Supports both string placeholders (using {placeholder_name} syntax) and 
    message list placeholders (using Placeholder objects).
    """
    
    def __init__(
        self,
        templates: Optional[List[Message]] = None,
        partial_variables: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ChatTemplate.
        
        Args:
            templates: List of message templates
            partial_variables: Default values for template variables
        """
        self.templates = templates or []
        self.messages: List[Message] = []
        self.partial_variables = partial_variables or {}
    
    @staticmethod
    def _convert_to_basemessage(item: Union[Tuple[str, str], str, BaseMessage]) -> BaseMessage:
        """Convert various message formats into BaseMessage objects."""
        if isinstance(item, tuple):
            role, content = item
            role = role.lower().strip()
            
            if role == "system":
                return SystemMessage(content=content)
            elif role == "user":
                return UserMessage(content=content)
            elif role in ["assistant", "ai"]:
                return AIMessage(content=content)
            elif role == "tool":
                return ToolMessage(content=content)
            elif role == "function":
                return FunctionMessage(content=content)
            else:
                raise ValueError(f"Unknown role: '{role}'")
        elif isinstance(item, str):
            return UserMessage(content=item)
        elif isinstance(item, BaseMessage):
            return item
        else:
            raise TypeError(f"Unsupported message type: {type(item)}")

    def run(self, **kwargs) -> List[BaseMessage]:
        """
        Format messages using templates and variables.
        
        Args:
            **kwargs: Template variables for formatting
            
        Returns:
            List of formatted BaseMessage objects
        """
        formatted_messages = []
        
        # Combine partial variables and provided variables
        all_variables = {**self.partial_variables, **kwargs}
        
        # Separate message lists from string values
        message_lists = {}
        string_variables = {}
        
        for key, value in all_variables.items():
            if isinstance(value, list) and all(isinstance(item, BaseMessage) for item in value):
                message_lists[key] = value
            else:
                string_variables[key] = str(value)
        
        # Process all messages
        all_messages = self.templates + self.messages

        for message in all_messages:
            if isinstance(message, Placeholder):
                # Insert message list
                if message.name in message_lists:
                    formatted_messages.extend(message_lists[message.name])
                else:
                    raise ValueError(f"Missing message list for placeholder: '{message.name}'")
            elif isinstance(message, BaseMessage):
                # Format string content
                try:
                    formatted_content = message.content.format(**string_variables)
                    new_message = message.__class__(content=formatted_content)
                    formatted_messages.append(new_message)
                except KeyError as e:
                    raise ValueError(f"Missing template variable: {e}")
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        
        return formatted_messages
    
    def add_message(self, role: str, content: str) -> 'ChatTemplate':
        """Add a message to the template."""
        role = role.lower().strip()
        
        if role == "system":
            message_obj = SystemMessage(content=content)
        elif role == "user":
            message_obj = UserMessage(content=content)
        elif role in ["assistant", "ai"]:
            message_obj = AIMessage(content=content)
        elif role == "tool":
            message_obj = ToolMessage(content=content)
        elif role == "function":
            message_obj = FunctionMessage(content=content)
        else:
            raise ValueError(f"Unknown role: '{role}'")
        
        self.messages.append(message_obj)
        return self
    
    def add_placeholder(self, name: str) -> 'ChatTemplate':
        """Add a message list placeholder."""
        self.messages.append(Placeholder(name))
        return self
    
    def partial(self, **kwargs) -> 'ChatTemplate':
        """
        Create a new template with partial variables filled.
        
        Args:
            **kwargs: Partial variables to set
            
        Returns:
            New ChatTemplate with updated partial variables
        """
        new_template = copy.deepcopy(self)
        new_template.partial_variables.update(kwargs)
        return new_template
    
    @classmethod
    def from_template(cls, templates: List[Union[Message, str]]) -> 'ChatTemplate':
        """Create template from a list of messages."""
        converted_templates = []
        for t in templates:
            if isinstance(t, Placeholder):
                converted_templates.append(t)
            else:
                converted_templates.append(cls._convert_to_basemessage(t))
        
        return cls(converted_templates)
    
    @classmethod 
    def from_messages(cls, *messages: Union[Message, str]) -> 'ChatTemplate':
        """Create template from individual message arguments."""
        return cls.from_template(list(messages))
    
    def copy(self) -> 'ChatTemplate':
        """Create a deep copy."""
        return copy.deepcopy(self)
    
    def __len__(self) -> int:
        return len(self.templates) + len(self.messages)
    
    def __repr__(self) -> str:
        return f"ChatTemplate(templates={len(self.templates)}, messages={len(self.messages)})"