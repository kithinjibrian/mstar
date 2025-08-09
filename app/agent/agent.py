from datetime import date
import time
import logging
from typing import Any, Dict, List, Generator, Callable
from uuid import uuid4

from app.ToT import ToT
from app.utils import (
    SystemMessage, UserMessage, AIMessage, FunctionMessage, BaseMessage,
    ChatTemplate, Placeholder, LLM
)
from .schema import FinalResponse, Replan, JoinOutputs
from .graph import Graph, END
from .parser import Parser, Task
from .register import Register

from .event import (
    EventBus,
    EventType
)

from .exec import (
    TaskExecutor,
    TaskStatus
)

# Configure logging
logging.basicConfig(level=None)
logger = logging.getLogger(__name__)

class Agent:    
    def __init__(
        self,
        tools: List = None,
        max_workers: int = 4,
        max_retries: int = 2,
        enable_streaming: bool = True,
        use_tot: bool = False
    ):
        self.id = str(uuid4())
        self.tools = tools or []
        self.enable_streaming = enable_streaming
        
        # Observability components
        self.event_bus = EventBus(self.id)
        self.task_executor = TaskExecutor(self.event_bus, max_workers, max_retries)
        
        # Agent components
        self.messages: List[BaseMessage] = []
        self.parser = Parser(self.tools)
        self.llm = LLM(model="groq/moonshotai/kimi-k2-instruct", temperature=1)
        self.joiner_llm = LLM(model="groq/moonshotai/kimi-k2-instruct", temperature=1)
        self.graph = self._setup_graph()

        self.use_tot = use_tot
        self.tot = ToT()
        
        # Setup templates
        self._setup_templates()
        
        # Register agent
        Register().set(self.id, self)
    
    def _setup_templates(self):
        """Setup improved prompt templates"""
        self.base_prompt = ChatTemplate.from_template([
            ("system", """You are an intelligent task planner that creates execution plans optimized for maximum parallelization. Your goal is to break down user requests into actionable tasks that can run concurrently whenever possible.

    ## Available Actions

    You have access to {num_tools} action types:

    {tool_descriptions}
    {num_tools}. join(defer_to_replanner: Union[str, NoneType]) – Synthesizes results from all previous actions. Required as the final step.

    Current date: {current_date}

    {replan_context}

    ## Planning Strategy

    Information Flow: Design your plan to gather data first, process it concurrently, then synthesize results.
    Parallelization: Identify tasks that can run simultaneously and execute them in parallel. Use dependencies only when necessary.
    Error Recovery: If some tasks might fail, plan alternative approaches or error handling strategies.
    Final Synthesis: Always end with join(defer_to_replanner: Union[str, NoneType]) to combine results and provide the final response.

    ## Technical Requirements

    1. Unique IDs: Each action must have a sequential ID (1, 2, 3, ...)
    2. Output References: Use $<id> to reference outputs from previous actions (e.g., $1, $2)
    3. Dependencies: Use 'wait_for' parameter when sequential execution is required
    4. Input Types: Actions accept either constants or outputs from previous actions
    5. Required Arguments: All action parameters are mandatory
    6. Plan Termination: End every plan with join(defer_to_replanner: Union[str, NoneType]) followed by <END_OF_PLAN>

    ## Deferring to Replanner

    When you cannot create a complete plan without seeing intermediate results:
    - Use join(defer_to_replanner="detailed explanation")
    - Explain why deferral is necessary
    - Describe expected outcomes and follow-up plans
    - Be specific about what information you need
    - Be verbose here, (e.g. include example plans, expected outcomes...). You goal is to help the replanner bootstrap.
             
    ## Output Format

    Respond ONLY with the task list in this exact format (Don't include any textual explanation):
    1. action_name(param1=value1, param2=value2)
    2. action_name(param1=value1, param2=$1)
    3. action_name(param1=value1, wait_for='$1 $2')
    ...
    N. join()
    <END_OF_PLAN>

    {task_counting_instruction}"""),
            Placeholder(name="messages"),
            ("system", "Generate the optimal execution plan now. Focus on maximizing parallelization while ensuring logical task dependencies.")
        ])
        
        self.joiner_prompt = ChatTemplate.from_template([
            ("system", """You are the execution orchestrator responsible for analyzing completed tasks and determining next steps.

    ## Your Responsibilities

    Task Analysis: Review all task results to understand what was accomplished and what failed.

    Decision Making: Choose the appropriate action based on the current state:
    - FinalResponse: When the user's request has been fully satisfied
    - Replan: When additional work, error recovery, or different approaches are needed

    Quality Assessment: Ensure the user receives a complete, accurate, and helpful response.

    ## Decision Criteria

    ### Choose FinalResponse when:
    - All required tasks completed successfully
    - User's request has been fully addressed
    - Results are sufficient and accurate
    - No critical errors that affect the outcome

    ### Choose Replan when:
    - Critical tasks failed and affect the final result
    - Missing information needed to complete the request
    - Results are incomplete or insufficient
    - Alternative approaches might yield better results
    - User's requirements aren't fully met

    ## Error Handling Guidelines

    Failure Analysis: For failed tasks, determine:
    - Root cause of the failure
    - Impact on overall objectives
    - Whether recovery is possible
    - Alternative approaches available

    Recovery Strategy: When replanning:
    - Provide specific, actionable feedback
    - Suggest concrete steps for error resolution
    - Identify which tasks need to be retried or replaced
    - Focus on successful completion paths

    ## Response Quality

    Completeness: Ensure responses fully address the user's original request.
    Accuracy: Verify information is correct and properly synthesized.
    Clarity: Present results in a clear, well-organized manner.
    Context: Maintain awareness of the user's intent and expectations."""),
            Placeholder(name="messages"),
            ("system", """Review the task execution results above and make your decision:

    1. Analyze what was accomplished successfully
    2. Identify any failures or gaps
    3. Determine if the user's request has been satisfied
    4. Choose FinalResponse (if complete) or Replan (if more work needed)
    5. Provide clear reasoning for your decision""")
        ])
    
    def _setup_graph(self) -> Graph:
        """Setup execution graph"""
        graph = Graph()
        graph.add_node("plan", self.plan)
        graph.add_node("join", self.joiner)
        
        graph.set_entry_point("plan")
        graph.add_edge("plan", "join")
        
        def should_continue(state) -> str:
            last_message = self.messages[-1] if self.messages else None
            if isinstance(last_message, AIMessage):
                return END
            return "plan"
        
        graph.add_conditional_edge("join", should_continue, {"plan": "plan", END: END})
        return graph
    
    def subscribe_to_events(self, event_type: EventType, callback: Callable):
        """Subscribe to specific events"""
        self.event_bus.subscribe(event_type, callback)
    
    def subscribe_to_all_events(self, callback: Callable):
        """Subscribe to all events"""
        self.event_bus.subscribe_all(callback)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current execution state for UI"""
        return {
            "agent_id": self.id,
            "execution_state": self.task_executor.execution_state.to_dict(),
            "message_count": len(self.messages),
            "tools_available": len(self.tools)
        }
    
    def plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a task plan by streaming from an LLM, submitting tasks for
        execution as soon as they are parsed. This enables concurrent planning
        and execution.
        """
        self.task_executor.execution_state.phase = "planning"
        self.event_bus.emit(EventType.PLANNING_STARTED)

        try:
            if any(isinstance(msg, SystemMessage) for msg in self.messages[-3:]):
                self.event_bus.emit(EventType.REPLAN_TRIGGERED)
                self.task_executor.reset_for_replan()

            observations = self._get_observations()
            num_tasks_submitted = 0
            logger.info("Starting LLM stream to generate and execute plan...")

            for task in self._create_planner():
                if self.task_executor.execution_state.execution_stopped:
                    logger.warning("Execution stopped due to a task failure or manual request. Halting task submission.")
                    break
                
                self.task_executor.submit_task(task, observations)
                num_tasks_submitted += 1
            
            logger.info(f"LLM streaming finished. A total of {num_tasks_submitted} tasks were submitted.")
            self.event_bus.emit(EventType.PLANNING_COMPLETED, plan_size=num_tasks_submitted)

            if num_tasks_submitted > 0:
                self.task_executor.execution_state.phase = "executing"
                logger.info("Waiting for all in-flight tasks to complete...")
                self.task_executor.wait_for_completion()
                logger.info("All tasks have finished execution.")
                self.event_bus.emit(EventType.PLANNING_COMPLETED)

            self._create_function_messages()

        except Exception as e:
            logger.error(f"An error occurred during the plan/execute phase: {e}", exc_info=True)
            self.task_executor.stop_execution(reason=f"agent_error: {e}")
            self.event_bus.emit(EventType.PLANNING_FAILED, error=str(e))
            self.messages.append(AIMessage(content=f"Planning or execution failed: {str(e)}"))

        return state
    
    def _create_planner(self) -> Generator[Task, None, None]:
        """Create streaming planner with observability"""
        if not self.tools:
            return
        
        tool_descriptions = "\n".join(
            f"{i + 1}. {getattr(tool, '_tool_schema', str(tool))}"
            for i, tool in enumerate(self.tools)
        )
        
        # Check for replanning
        should_replan = any(isinstance(msg, SystemMessage) for msg in self.messages[-3:])
        
        if should_replan:
            next_task_idx = self._get_next_task_index()
            replan_context = """REPLANNING MODE:
- Use previous plan results to create an improved plan
- Never repeat executed actions
- Continue task numbering from previous plan
- Focus on error recovery and completion"""
            task_instruction = f"IMPORTANT: Begin task numbering at {next_task_idx}"
        else:
            replan_context = ""
            task_instruction = ""
        
        prompt = self.base_prompt.partial(
            num_tools=len(self.tools) + 1,
            tool_descriptions=tool_descriptions,
            current_date=date.today(),
            replan_context=replan_context,
            task_counting_instruction=task_instruction
        )
        
        try:
            msgs = prompt.run(messages=self.messages)

            stream = self.llm.stream(msgs)
            
            self.parser.reset_stream()
            
            for chunk in stream:
                #print(chunk, end="", flush=True)
                self.event_bus.emit(
                    EventType.PLANNING_STREAM_CHUNK,
                    chunk=chunk
                )
                
                yield from self.parser.stream(chunk)
            
            yield from self.parser.finalize_stream()
            
        except Exception as e:
            logger.error(f"Streaming planning failed: {e}")
    
    def _process_completed_tasks(self, completed_task_ids: List[int], observations: Dict[int, Any]):
        """Process completed tasks and update observations"""
        for task_id in completed_task_ids:
            task_state = self.task_executor.execution_state.current_tasks.get(task_id)
            if task_state and task_state.status == TaskStatus.COMPLETED:
                observations[task_id] = task_state.result
    
    def _create_function_messages(self):
        """Create function messages from task executions"""
        for task_id in sorted(self.task_executor.execution_state.current_tasks.keys()):
            task_state = self.task_executor.execution_state.current_tasks[task_id]
            
            if task_state.status == TaskStatus.COMPLETED:
                content = str(task_state.result)
            elif task_state.status == TaskStatus.FAILED:
                content = f"FAILED: {task_state.error}"
            elif task_state.status == TaskStatus.CANCELLED:
                content = f"CANCELLED: Task cancelled due to upstream failure"
            else:
                content = f"NOT_EXECUTED: Task status {task_state.status.value}"
            
            metadata = {
                "idx": task_id,
                "args": task_state.args,
                "status": task_state.status.value,
                "duration": task_state.duration_ms,
                "observable_execution": True
            }
            
            if task_state.error:
                metadata["error"] = task_state.error
            
            function_msg = FunctionMessage(
                name=task_state.tool_name,
                content=content,
                metadata=metadata,
                tool_call_id=str(task_id)
            )
            
            self.messages.append(function_msg)
    
    def joiner(self, state):
        """Joining phase with observability"""
        try:
            self.task_executor.execution_state.phase = "joining"
            self.event_bus.emit(EventType.JOINING_STARTED)
            
            recent_messages = self._select_recent_messages()
            
            self.event_bus.emit(EventType.JOINING_THINKING)
            decision = self._run_joiner(recent_messages)
            
            new_messages = self._parse_joiner_output(decision)
            self.messages.extend(new_messages)
            
            self.event_bus.emit(EventType.JOINING_COMPLETED)
            
        except Exception as e:
            self.event_bus.emit(EventType.JOINING_FAILED, error=str(e))
            self.messages.append(AIMessage(content=f"Error in execution: {str(e)}"))
        
        return self.messages
    
    def _run_joiner(self, messages: List[BaseMessage]) -> JoinOutputs:
        """Run joiner with structured output"""
        try:
            jllm = self.joiner_llm.with_structured_output(JoinOutputs)
            msgs = self.joiner_prompt.run(messages=messages)
            result = jllm.run(msgs)
            
            return result.content if hasattr(result, 'content') else result
            
        except Exception as e:
            logger.error(f"Joiner execution failed: {e}")
            return JoinOutputs(
                thought="Error occurred in joiner execution",
                action=FinalResponse(response=f"An error occurred while processing: {str(e)}")
            )
    
    def _parse_joiner_output(self, decision: JoinOutputs) -> List[BaseMessage]:
        """Parse joiner output into messages"""
        try:
            response = [AIMessage(content=f"Thought: {decision.thought}")]
            
            if isinstance(decision.action, Replan):
                self.event_bus.emit(
                    EventType.REPLAN_CONTEXT_CREATED,
                    feedback=decision.action.feedback
                )
                response.append(SystemMessage(content=f"Context from last attempt: {decision.action.feedback}"))
            else:
                # Emit AI response completion
                self.event_bus.emit(
                    EventType.AI_RESPONSE_COMPLETED,
                    response=decision.action.response
                )
                response.append(AIMessage(content=decision.action.response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error parsing joiner output: {e}")
            return [AIMessage(content=f"Error in processing: {str(e)}")]
    
    def _select_recent_messages(self, max_messages: int = 50) -> List[BaseMessage]:
        """Select recent messages for context"""
        selected = []
        for msg in reversed(self.messages):
            selected.append(msg)
            if isinstance(msg, UserMessage) or len(selected) >= max_messages:
                break
        selected.reverse()
        return selected
    
    def _get_next_task_index(self) -> int:
        """Get next task index from message history"""
        next_task = 1
        for message in reversed(self.messages):
            if isinstance(message, FunctionMessage) and "idx" in message.metadata:
                try:
                    next_task = message.metadata["idx"] + 1
                    break
                except (ValueError, KeyError):
                    continue
        return next_task
    
    def _get_observations(self) -> Dict[int, Any]:
        """Extract observations from message history"""
        observations = {}
        for message in reversed(self.messages):
            if isinstance(message, FunctionMessage) and "idx" in message.metadata:
                try:
                    idx = int(message.metadata["idx"])
                    if not message.content.startswith("FAILED:"):
                        observations[idx] = message.content
                except (ValueError, KeyError):
                    continue
        return observations
    
    def run(self, prompt: str) -> None:
        """Main execution method with full observability"""
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided")
            return
        
        try:

            if self.use_tot:
                result = self.tot.solve(prompt.strip())
                enhanced_prompt = f"Create a plan that closely follows this steps: {result.get("final_solution")}"

                print(enhanced_prompt)
                
                self.messages.append(UserMessage(content=enhanced_prompt))
            else:
                self.messages.append(UserMessage(content=prompt.strip()))
            
            self.event_bus.emit(
                EventType.AGENT_STARTED,
                prompt=prompt[:200],  # Truncate for logging
                tools_count=len(self.tools)
            )
            
            start_time = time.time()
            self.graph.execute()
            execution_time = time.time() - start_time

            print(self.messages)
            
            self.event_bus.emit(
                EventType.AGENT_COMPLETED,
                execution_time=execution_time,
                total_messages=len(self.messages)
            )
            
            logger.info(f"Agent execution completed in {execution_time:.2f}s")
            
        except Exception as e:
            self.event_bus.emit(
                EventType.AGENT_FAILED,
                error=str(e)
            )
            logger.error(f"Agent execution failed: {e}")
            self.messages.append(AIMessage(content=f"Execution failed: {str(e)}"))
        finally:
            self.task_executor.shutdown()