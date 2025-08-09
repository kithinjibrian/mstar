from litellm import completion, acompletion
from pydantic import BaseModel
import json
from typing import Optional, Dict, Any, List, Iterator, AsyncIterator, Union
from .message import (
    SystemMessage,
    UserMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    FunctionMessage
)

class StreamingResponse:
    """Wrapper for streaming response chunks"""
    def __init__(self, chunk_iterator: Iterator[Dict], model: str, structured_schema: Optional[BaseModel] = None):
        self.chunk_iterator = chunk_iterator
        self.model = model
        self.structured_schema = structured_schema
        self._accumulated_content = ""
        self._tool_calls = []
        self._metadata = {
            "model": model,
            "usage": {},
            "finish_reason": None,
            "tool_calls": []
        }
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over content chunks"""
        for chunk in self.chunk_iterator:
            try:
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                reasoning_content = delta.get("reasoning_content", "")

                if content:
                    print(content, end="", flush=True)
                    self._accumulated_content += content
                    yield content
                
                # Handle tool calls in streaming
                if delta.get("tool_calls"):
                    for tool_call_delta in delta["tool_calls"]:
                        index = tool_call_delta.get("index", 0)
                        
                        # Initialize tool call if needed
                        while len(self._tool_calls) <= index:
                            self._tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        # Update tool call
                        if "id" in tool_call_delta:
                            self._tool_calls[index]["id"] += tool_call_delta["id"]
                        
                        if "function" in tool_call_delta:
                            func_delta = tool_call_delta["function"]
                            if "name" in func_delta:
                                self._tool_calls[index]["function"]["name"] += func_delta["name"]
                            if "arguments" in func_delta:
                                self._tool_calls[index]["function"]["arguments"] += func_delta["arguments"]
                
                # Update metadata from final chunk
                if choice.get("finish_reason"):
                    self._metadata["finish_reason"] = choice["finish_reason"]
                
                # Update usage if present
                if chunk.get("usage"):
                    self._metadata["usage"] = chunk["usage"]
                    
            except Exception as e:
                print(f"Warning: Error processing stream chunk: {e}")
                continue
    
    def get_final_message(self) -> AIMessage:
        """Get the complete accumulated message after streaming"""
        content = self._accumulated_content
        
        # Parse structured output if applicable
        if self.structured_schema and content:
            try:
                parsed_json = json.loads(content)
                structured_content = self.structured_schema(**parsed_json)
                content = structured_content
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to parse structured output from stream: {e}")
        
        # Update metadata with accumulated tool calls
        if self._tool_calls:
            self._metadata["tool_calls"] = self._tool_calls
        
        return AIMessage(
            content=content,
            tool_calls=self._tool_calls if self._tool_calls else None,
            metadata=self._metadata
        )

class AsyncStreamingResponse:
    """Async wrapper for streaming response chunks"""
    def __init__(self, chunk_iterator: AsyncIterator[Dict], model: str, structured_schema: Optional[BaseModel] = None):
        self.chunk_iterator = chunk_iterator
        self.model = model
        self.structured_schema = structured_schema
        self._accumulated_content = ""
        self._tool_calls = []
        self._metadata = {
            "model": model,
            "usage": {},
            "finish_reason": None,
            "tool_calls": []
        }
    
    async def __aiter__(self) -> AsyncIterator[str]:
        """Async iterate over content chunks"""
        async for chunk in self.chunk_iterator:
            try:
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    self._accumulated_content += content
                    yield content
                
                # Handle tool calls in streaming
                if delta.get("tool_calls"):
                    for tool_call_delta in delta["tool_calls"]:
                        index = tool_call_delta.get("index", 0)
                        
                        # Initialize tool call if needed
                        while len(self._tool_calls) <= index:
                            self._tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        # Update tool call
                        if "id" in tool_call_delta:
                            self._tool_calls[index]["id"] += tool_call_delta["id"]
                        
                        if "function" in tool_call_delta:
                            func_delta = tool_call_delta["function"]
                            if "name" in func_delta:
                                self._tool_calls[index]["function"]["name"] += func_delta["name"]
                            if "arguments" in func_delta:
                                self._tool_calls[index]["function"]["arguments"] += func_delta["arguments"]
                
                # Update metadata from final chunk
                if choice.get("finish_reason"):
                    self._metadata["finish_reason"] = choice["finish_reason"]
                
                # Update usage if present
                if chunk.get("usage"):
                    self._metadata["usage"] = chunk["usage"]
                    
            except Exception as e:
                print(f"Warning: Error processing async stream chunk: {e}")
                continue
    
    async def get_final_message(self) -> AIMessage:
        """Get the complete accumulated message after streaming"""
        content = self._accumulated_content
        
        # Parse structured output if applicable
        if self.structured_schema and content:
            try:
                parsed_json = json.loads(content)
                structured_content = self.structured_schema(**parsed_json)
                content = structured_content
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Failed to parse structured output from async stream: {e}")
        
        # Update metadata with accumulated tool calls
        if self._tool_calls:
            self._metadata["tool_calls"] = self._tool_calls
        
        return AIMessage(
            content=content,
            tool_calls=self._tool_calls if self._tool_calls else None,
            metadata=self._metadata
        )

class LLM:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.default_params = kwargs
        self._structured_output_schema = None
    
    def with_structured_output(self, base_model: BaseModel):
        """Return a new LLM instance configured for structured output"""
        new_llm = LLM(self.model, **self.default_params)
        new_llm._structured_output_schema = base_model
        return new_llm
    
    def _prepare_messages_and_params(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None, **kwargs) -> tuple[List[Dict], Dict]:
        """Prepare messages and parameters for API call"""
        # Convert messages to LiteLLM format
        msgs = [self.convert(message) for message in messages]
        
        # Prepare parameters
        params = {**self.default_params, **kwargs}
        
        # Add tools if provided
        if tools:
            params['tools'] = tools
            # Set tool_choice if not explicitly provided
            if 'tool_choice' not in params:
                params['tool_choice'] = 'auto'
        
        # Handle structured output
        if self._structured_output_schema:
            params['response_format'] = {
                "type": "json_object"
            }
            # Add schema instruction to system message or create one
            schema_instruction = f"Respond with valid JSON that matches this schema: {self._structured_output_schema.model_json_schema()}"
            
            # Find system message or add one
            system_msg_found = False
            for msg in msgs:
                if msg["role"] == "system":
                    msg["content"] += f"\n\n{schema_instruction}"
                    system_msg_found = True
                    break
            
            if not system_msg_found:
                msgs.insert(0, {"role": "system", "content": schema_instruction})
        
        return msgs, params
    
    def run(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None, **kwargs) -> AIMessage:
        """Run the LLM with the given messages"""
        try:
            msgs, params = self._prepare_messages_and_params(messages, tools, **kwargs)
            
            # Make API call
            response = completion(
                model=self.model,
                messages=msgs,
                **params
            )
            
            choice = response["choices"][0]["message"]
            content = choice.get("content", "")
            tool_calls = choice.get("tool_calls", [])
            
            # Parse structured output if applicable
            if self._structured_output_schema and content:
                try:
                    parsed_json = json.loads(content)
                    structured_content = self._structured_output_schema(**parsed_json)
                    content = structured_content
                except (json.JSONDecodeError, ValueError) as e:
                    # Fallback to raw content if parsing fails
                    print(f"Warning: Failed to parse structured output: {e}")
            
            return AIMessage(
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                metadata={
                    "model": self.model,
                    "usage": response.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": tool_calls
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"LLM completion failed: {str(e)}") from e
    
    def stream(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None, **kwargs) -> StreamingResponse:
        """Stream the LLM response"""
        try:
            msgs, params = self._prepare_messages_and_params(messages, tools, **kwargs)
            
            # Force streaming
            params['stream'] = True
            
            # Make streaming API call
            response = completion(
                model=self.model,
                messages=msgs,
                **params
            )
            
            return StreamingResponse(
                chunk_iterator=response,
                model=self.model,
                structured_schema=self._structured_output_schema
            )
            
        except Exception as e:
            raise RuntimeError(f"LLM streaming failed: {str(e)}") from e
    
    async def arun(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None, **kwargs) -> AIMessage:
        """Async version of run"""
        try:
            msgs, params = self._prepare_messages_and_params(messages, tools, **kwargs)
            
            response = await acompletion(
                model=self.model,
                messages=msgs,
                **params
            )
            
            choice = response["choices"][0]["message"]
            content = choice.get("content", "")
            tool_calls = choice.get("tool_calls", [])
            
            if self._structured_output_schema and content:
                try:
                    parsed_json = json.loads(content)
                    structured_content = self._structured_output_schema(**parsed_json)
                    content = structured_content
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Failed to parse structured output: {e}")
            
            return AIMessage(
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                metadata={
                    "model": self.model,
                    "usage": response.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": tool_calls
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Async LLM completion failed: {str(e)}") from e
    
    async def astream(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None, **kwargs) -> AsyncStreamingResponse:
        """Async stream the LLM response"""
        try:
            msgs, params = self._prepare_messages_and_params(messages, tools, **kwargs)
            
            # Force streaming
            params['stream'] = True
            
            # Make async streaming API call
            response = await acompletion(
                model=self.model,
                messages=msgs,
                **params
            )
            
            return AsyncStreamingResponse(
                chunk_iterator=response,
                model=self.model,
                structured_schema=self._structured_output_schema
            )
            
        except Exception as e:
            raise RuntimeError(f"Async LLM streaming failed: {str(e)}") from e
    
    @staticmethod
    def convert(msg: BaseMessage) -> Dict[str, Any]:
        """Convert custom message types to LiteLLM format"""
        if isinstance(msg, SystemMessage):
            return {
                "role": "system",
                "content": msg.content
            }
        elif isinstance(msg, UserMessage):
            return {
                "role": "user",
                "content": msg.content
            }
        elif isinstance(msg, (AIMessage, FunctionMessage)):
            result = {
                "role": "assistant",
                "content": msg.content or ""
            }
            # Add tool calls if present
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                result["tool_calls"] = msg.tool_calls
            return result
        elif isinstance(msg, ToolMessage):
            result = {
                "role": "tool",
                "content": msg.content
            }
            # Add tool_call_id if present
            if hasattr(msg, 'tool_call_id'):
                result["tool_call_id"] = msg.tool_call_id
            # Add name if present
            if hasattr(msg, 'name'):
                result["name"] = msg.name
            return result
        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")