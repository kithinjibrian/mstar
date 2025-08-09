import json
from typing import Any, Dict, List, Union, Iterator, AsyncIterator
from uuid import uuid4

from app.utils import (
    SystemMessage,
    UserMessage,
    AIMessage,
    BaseMessage,
    LLM,
    ToolMessage
)

from .register import Register

class ReAct:
    def __init__(
        self,
        stream: bool = True,
        tools: List = None,
        system_prompt: str = "You're a helpful AI assistant",
        model: str = "groq/moonshotai/kimi-k2-instruct",
        max_iterations: int = 10,
        coder: bool = False,
    ):
        self.id = uuid4()
        self.stream = stream
        self.coder = coder
        self.max_iterations = max_iterations        
        # Initialize tools
        self.tools = tools or []
        
        self.messages: List[BaseMessage] = [SystemMessage(
            content=self._build_system_prompt(system_prompt)
        )]

        # Initialize LLM without tools (tools handled by agent)
        self.llm = LLM(model=model)

        Register().set(self.id, self)
    
    def _build_system_prompt(self, base_prompt: str) -> str:
        """Build system prompt with ReAct instructions"""
        if not self.tools:
            return base_prompt
        
        tool_descriptions = "\n".join(
            f"{i + 1}. {getattr(tool, '_tool_schema', str(tool))}"
            for i, tool in enumerate(self.tools)
        )
        
        react_prompt = f"""{base_prompt}

You have access to the following tools:
{tool_descriptions}

When you need to use a tool, follow this pattern:
1. Think about what you need to do
2. Use the appropriate tool
3. Observe the result
4. Continue reasoning based on the result

You can use tools multiple times and combine their results to answer complex questions."""
        
        return react_prompt
    
    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolMessage]:
        """Execute tool calls and return tool messages"""
        results = []
        
        for tool_call in tool_calls:
            try:
                tool_id = tool_call["id"]
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                tool = None

                for _tool in self.tools:
                    if _tool.__name__ == function_name:
                        tool = _tool

                if tool:
                    error, result = tool(**arguments)
                    
                    if error:
                        results.append(ToolMessage(
                            content=f"Error: {result}",
                            tool_call_id=tool_id,
                            name=function_name
                        ))
                    else:
                        if not isinstance(result, str):
                            result = json.dumps(result)

                        results.append(ToolMessage(
                            content=result,
                            tool_call_id=tool_id,
                            name=function_name
                        ))
            
            except Exception as e:
                results.append(ToolMessage(
                    content=f"Error executing tool: {str(e)}",
                    tool_call_id=tool_call.get("id", "unknown"),
                    name=tool_call.get("function", {}).get("name", "unknown")
                ))
        
        return results
    
    def run(self, prompt: str) -> Union[AIMessage, Iterator[str]]:
        """Run the agent with the given prompt"""
        # Add user message
        self.messages.append(UserMessage(content=prompt))
        
        if self.stream:
            return self._run_streaming()
        else:
            return self._run_sync()
    
    def _run_sync(self) -> AIMessage:
        """Run agent synchronously"""
        iteration = 0
        
        while iteration < self.max_iterations:
            # Get LLM response with tools
            tools = [tool._openai_schema for tool in self.tools]
            response = self.llm.run(self.messages, tools=tools)
            self.messages.append(response)

            # print(response)
            
            # Check if we have tool calls
            if response.tool_calls:
                # Execute tool calls
                tool_results = self.execute_tool_calls(response.tool_calls)
                self.messages.extend(tool_results)
                
                # Continue to next iteration
                iteration += 1
                continue
            else:
                # No tool calls, we're done
                return response
        
        # If we hit max iterations, return the last response
        return response
    
    def _run_streaming(self) -> Iterator[str]:
        """Run agent with streaming"""
        iteration = 0
        
        while iteration < self.max_iterations:
            # Stream the response with tools
            tools = self._tool_definitions if self.tools else None
            stream_response = self.llm.stream(self.messages, tools=tools)
            
            # Yield content as it comes
            for chunk in stream_response:
                yield chunk
            
            # Get the final message after streaming
            final_message = stream_response.get_final_message()
            self.messages.append(final_message)
            
            # Check if we have tool calls
            if final_message.tool_calls:
                # Execute tool calls
                tool_results = self.execute_tool_calls(final_message.tool_calls)
                self.messages.extend(tool_results)
                
                # Yield indication that we're using tools
                yield f"\n\n[Using tools: {', '.join([tc['function']['name'] for tc in final_message.tool_calls])}]\n\n"
                
                # Continue to next iteration
                iteration += 1
                continue
            else:
                # No tool calls, we're done
                break
        
        if iteration >= self.max_iterations:
            yield "\n\n[Maximum iterations reached]"
    
    async def arun(self, prompt: str) -> Union[AIMessage, AsyncIterator[str]]:
        """Run the agent asynchronously"""
        # Add user message
        self.messages.append(UserMessage(content=prompt))
        
        if self.stream:
            return self._arun_streaming()
        else:
            return await self._arun_sync()
    
    async def _arun_sync(self) -> AIMessage:
        """Run agent asynchronously without streaming"""
        iteration = 0
        
        while iteration < self.max_iterations:
            # Get LLM response with tools
            tools = self._tool_definitions if self.tools else None
            response = await self.llm.arun(self.messages, tools=tools)
            self.messages.append(response)
            
            # Check if we have tool calls
            if response.tool_calls:
                # Execute tool calls
                tool_results = self.execute_tool_calls(response.tool_calls)
                self.messages.extend(tool_results)
                
                # Continue to next iteration
                iteration += 1
                continue
            else:
                # No tool calls, we're done
                return response
        
        # If we hit max iterations, return the last response
        return response
    
    async def _arun_streaming(self) -> AsyncIterator[str]:
        """Run agent asynchronously with streaming"""
        iteration = 0
        
        while iteration < self.max_iterations:
            # Stream the response with tools
            tools = self._tool_definitions if self.tools else None
            stream_response = await self.llm.astream(self.messages, tools=tools)
            
            # Yield content as it comes
            async for chunk in stream_response:
                yield chunk
            
            # Get the final message after streaming
            final_message = await stream_response.get_final_message()
            self.messages.append(final_message)
            
            # Check if we have tool calls
            if final_message.tool_calls:
                # Execute tool calls
                tool_results = self.execute_tool_calls(final_message.tool_calls)
                self.messages.extend(tool_results)
                
                # Yield indication that we're using tools
                yield f"\n\n[Using tools: {', '.join([tc['function']['name'] for tc in final_message.tool_calls])}]\n\n"
                
                # Continue to next iteration
                iteration += 1
                continue
            else:
                # No tool calls, we're done
                break
        
        if iteration >= self.max_iterations:
            yield "\n\n[Maximum iterations reached]"
    
    def reset(self):
        """Reset the conversation, keeping only the system message"""
        system_msg = self.messages[0] if self.messages and isinstance(self.messages[0], SystemMessage) else None
        if system_msg:
            self.messages = [system_msg]
        else:
            self.messages = []
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation"""
        return self.messages.copy()
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt"""
        new_system_msg = SystemMessage(content=self._build_system_prompt(prompt))
        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = new_system_msg
        else:
            self.messages.insert(0, new_system_msg)