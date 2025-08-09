from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import queue
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from datetime import datetime, date
import weakref
import traceback
from contextlib import contextmanager

from .parser import Task
from .event import EventBus, EventType

logging.basicConfig(level=None)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"  
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # New status for when dependencies fail

@dataclass
class TaskState:
    """Complete state information for a task"""
    task_id: int
    name: str
    tool_name: str
    args: Dict[str, Any]
    dependencies: Set[int]
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    retry_count: int = 0
    future: Optional[Future] = field(default=None, repr=False)
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "tool_name": self.tool_name,
            "args": self.args,
            "dependencies": list(self.dependencies),
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "progress": self.progress,
            "retry_count": self.retry_count
        }

@dataclass
class ExecutionState:
    """Current execution state for UI visibility"""
    phase: str = "idle"  # idle, planning, executing, joining, replanning
    current_tasks: Dict[int, TaskState] = field(default_factory=dict)
    completed_tasks: Set[int] = field(default_factory=set)
    failed_tasks: Set[int] = field(default_factory=set)
    cancelled_tasks: Set[int] = field(default_factory=set)
    skipped_tasks: Set[int] = field(default_factory=set)
    total_tasks: int = 0
    execution_stopped: bool = False
    current_ai_response: str = ""
    observations: Dict[int, Any] = field(default_factory=dict)  # Store task results
    
    def to_dict(self) -> Dict[str, Any]:
        finished_count = len(self.completed_tasks) + len(self.failed_tasks) + len(self.cancelled_tasks) + len(self.skipped_tasks)
        return {
            "phase": self.phase,
            "current_tasks": {k: v.to_dict() for k, v in self.current_tasks.items()},
            "completed_count": len(self.completed_tasks),
            "failed_count": len(self.failed_tasks),
            "cancelled_count": len(self.cancelled_tasks),
            "skipped_count": len(self.skipped_tasks),
            "total_tasks": self.total_tasks,
            "execution_stopped": self.execution_stopped,
            "current_ai_response": self.current_ai_response,
            "progress_percentage": finished_count / max(1, self.total_tasks) * 100
        }

class TaskExecutionError(Exception):
    """Custom exception for task execution errors"""
    def __init__(self, task_id: int, message: str, original_error: Optional[Exception] = None):
        self.task_id = task_id
        self.original_error = original_error
        super().__init__(f"Task {task_id}: {message}")

class TaskExecutor:
    """Enhanced task executor with improved reliability and observability"""
    
    def __init__(self, event_bus: EventBus, max_workers: int = 4, max_retries: int = 2, 
                 fail_fast: bool = True, retry_delay: float = 0.5):
        self.event_bus = event_bus
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.fail_fast = fail_fast
        self.retry_delay = retry_delay
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TaskWorker")
        self.execution_state = ExecutionState()
        self.result_queue = queue.Queue()
        
        self._shutdown = False
        self._lock = threading.RLock()  # Use RLock to avoid deadlocks
        self._task_registry: Dict[int, Task] = {}  # Store original tasks
        
        # Metrics
        self._execution_start_time: Optional[datetime] = None
        self._task_completion_callbacks: List[Callable[[int, TaskStatus, Any], None]] = []
        
    def add_completion_callback(self, callback: Callable[[int, TaskStatus, Any], None]):
        """Add a callback for task completion events"""
        self._task_completion_callbacks.append(callback)
        
    def submit_task(self, task: Task, observations: Optional[Dict[int, Any]] = None) -> bool:
        """Submit a task for execution, returns True if submitted"""
        if self.execution_state.execution_stopped:
            logger.warning(f"Cannot submit task {task.get('idx')}: execution stopped")
            return False
            
        task_id = task["idx"]
        dependencies = set(task.get("dependencies", []))
        tool = task["tool"]
        tool_name = getattr(tool, '__name__', str(tool))
        
        # Validate dependencies exist
        if observations:
            missing_deps = dependencies - set(observations.keys()) - set(self._task_registry.keys())
            if missing_deps:
                logger.error(f"Task {task_id} has missing dependencies: {missing_deps}")
                return False
        
        # Create task state
        task_state = TaskState(
            task_id=task_id,
            name=task.get("name", f"task_{task_id}"),
            tool_name=tool_name,
            args=task.get("args", {}),
            dependencies=dependencies,
            status=TaskStatus.PENDING
        )
        
        with self._lock:
            self._task_registry[task_id] = task
            self.execution_state.current_tasks[task_id] = task_state
            self.execution_state.total_tasks += 1
            
            if observations:
                self.execution_state.observations.update(observations)
        
        self.event_bus.emit(
            EventType.TASK_QUEUED,
            task_id=task_id,
            task_name=task_state.name,
            tool_name=tool_name,
            dependencies=list(dependencies)
        )

        logger.debug(f"Submitted task '{task_state.name}' (ID: {task_id})")
        
        # Check if task is ready to execute immediately
        if self._is_task_ready(task_state):
            self._execute_task(task_id)
        else:
            task_state.status = TaskStatus.QUEUED
            
        return True
    
    def _is_task_ready(self, task_state: TaskState) -> bool:
        """Check if a task's dependencies are satisfied"""
        return (task_state.dependencies.issubset(self.execution_state.completed_tasks) and
                not task_state.dependencies.intersection(self.execution_state.failed_tasks))
    
    def _execute_task(self, task_id: int):
        """Execute a single task"""
        if self.execution_state.execution_stopped:
            return
            
        with self._lock:
            task_state = self.execution_state.current_tasks.get(task_id)
            task = self._task_registry.get(task_id)
            
            if not task_state or not task:
                logger.error(f"Task {task_id} not found in registry")
                return
                
            # Check if any dependencies failed
            failed_deps = task_state.dependencies.intersection(self.execution_state.failed_tasks)
            if failed_deps:
                self._mark_task_skipped(task_state, f"Dependencies failed: {failed_deps}")
                return
            
            task_state.status = TaskStatus.RUNNING
            task_state.start_time = datetime.now()
        
        self.event_bus.emit(
            EventType.TASK_STARTED,
            task_id=task_state.task_id,
            task_name=task_state.name,
            tool_name=task_state.tool_name
        )
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_with_retry, task_id)
        
        # Store future reference
        with self._lock:
            task_state.future = future
        
        # Add completion callback
        future.add_done_callback(
            lambda f: self._on_task_complete(task_id, f)
        )
    
    def _execute_with_retry(self, task_id: int) -> Tuple[bool, Any]:
        """Execute task with retry logic"""
        with self._lock:
            task_state = self.execution_state.current_tasks[task_id]
            task = self._task_registry[task_id]
        
        for attempt in range(self.max_retries + 1):
            if self.execution_state.execution_stopped:
                return (True, "Execution stopped")
                
            try:
                with self._lock:
                    task_state.retry_count = attempt
                
                self.event_bus.emit(
                    EventType.TASK_PROGRESS,
                    task_id=task_id,
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1
                )
                
                return self._execute_single_task(task, self.execution_state.observations)
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                logger.warning(f"Task {task_id} failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** attempt) * (0.5 + 0.5 * hash(task_id) % 100 / 100)
                    time.sleep(delay)
                else:
                    return (True, f"{error_msg}\nFinal attempt failed after {self.max_retries + 1} attempts")
        
        return (True, "Max retries exceeded")
    
    def _execute_single_task(self, task: Task, observations: Dict[int, Any]) -> Any:
        """Execute a single task instance with better error handling"""
        tool = task["tool"]
        
        if isinstance(tool, str):
            if tool == "join":
                if task['args']:
                    return (
                        False,
                        f"""The planner deferred this task to you. 
{json.dumps(task['args'])}
"""
                    )
            return (False, tool)
        
        try:
            # Resolve arguments
            args = task.get("args", {})
            resolved_args = self._resolve_arguments(args, observations)

            logger.info(f"Executing task '{task['name']}' (Args: {resolved_args})")
            
            # Validate tool is callable
            if not callable(tool):
                raise TaskExecutionError(task["idx"], f"Tool {tool} is not callable")
            
            # Execute tool with proper argument handling
            if isinstance(resolved_args, dict):
                return tool(**resolved_args)
            elif isinstance(resolved_args, (list, tuple)):
                return tool(*resolved_args)
            else:
                return tool(resolved_args)
                
        except Exception as e:
            # Preserve original exception with traceback
            raise TaskExecutionError(
                task["idx"], 
                f"Tool execution failed: {str(e)}", 
                original_error=e
            ) from e
    
    def _resolve_arguments(self, args: Any, observations: Dict[int, Any]) -> Any:
        """Resolve task arguments with dependency substitution and better error handling"""
        ID_PATTERN = r"\$\{?(\d+)\}?"
        
        def replace_match(match) -> str:
            idx = int(match.group(1))
            if idx not in observations:
                available_ids = sorted(observations.keys())
                raise ValueError(f"Referenced task {idx} not found. Available: {available_ids}")
            
            result = observations[idx]
            # Handle None results gracefully
            return str(result) if result is not None else ""
        
        try:
            if isinstance(args, str):
                return re.sub(ID_PATTERN, replace_match, args)
            elif isinstance(args, list):
                return [self._resolve_arguments(a, observations) for a in args]
            elif isinstance(args, dict):
                return {k: self._resolve_arguments(v, observations) for k, v in args.items()}
            else:
                return args
        except Exception as e:
            raise ValueError(f"Failed to resolve arguments {args}: {e}") from e
    
    def _mark_task_skipped(self, task_state: TaskState, reason: str):
        """Mark a task as skipped due to failed dependencies"""
        with self._lock:
            task_state.status = TaskStatus.SKIPPED
            task_state.error = reason
            task_state.end_time = datetime.now()
            self.execution_state.skipped_tasks.add(task_state.task_id)
        
        self.event_bus.emit(
            EventType.TASK_CANCELLED,  # Reuse existing event type
            task_id=task_state.task_id,
            task_name=task_state.name,
            reason=reason
        )
        
        logger.info(f"Task '{task_state.name}' skipped: {reason}")
        self.result_queue.put(task_state.task_id)
    
    def _on_task_complete(self, task_id: int, future: Future):
        """Handle task completion with improved error handling"""
        if self._shutdown:
            return
            
        with self._lock:
            task_state = self.execution_state.current_tasks.get(task_id)
            if not task_state:
                logger.error(f"Task state for {task_id} not found during completion")
                return
                
            task_state.end_time = datetime.now()
            task_state.future = None  # Clear future reference
            
            try:
                error_flag, result = future.result()
                
                if not error_flag:
                    # Success
                    task_state.result = result
                    task_state.status = TaskStatus.COMPLETED
                    self.execution_state.completed_tasks.add(task_id)
                    self.execution_state.observations[task_id] = result
                    
                    logger.info(f"Task '{task_state.name}' completed successfully in {task_state.duration_ms:.2f}ms")
                    
                    self.event_bus.emit(
                        EventType.TASK_COMPLETED,
                        task_id=task_id,
                        task_name=task_state.name,
                        result=str(result),
                        duration_ms=task_state.duration_ms
                    )
                    
                    # Trigger completion callbacks
                    for callback in self._task_completion_callbacks:
                        try:
                            callback(task_id, TaskStatus.COMPLETED, result)
                        except Exception as e:
                            logger.error(f"Completion callback failed: {e}")
                else:
                    # Failure
                    task_state.status = TaskStatus.FAILED
                    task_state.error = str(result)
                    self.execution_state.failed_tasks.add(task_id)
                    
                    logger.error(f"Task '{task_state.name}' failed after {task_state.retry_count + 1} attempts: {result}")
                    
                    self.event_bus.emit(
                        EventType.TASK_FAILED,
                        task_id=task_id,
                        task_name=task_state.name,
                        error=str(result),
                        duration_ms=task_state.duration_ms
                    )
                    
                    # Trigger completion callbacks
                    for callback in self._task_completion_callbacks:
                        try:
                            callback(task_id, TaskStatus.FAILED, result)
                        except Exception as e:
                            logger.error(f"Completion callback failed: {e}")
                    
                    # Handle failure based on fail_fast setting
                    if self.fail_fast:
                        self.stop_execution("task_failure")
                    
            except Exception as e:
                # Unexpected exception during result processing
                logger.error(f"Unexpected error handling task {task_id} completion: {e}")
                logger.error(traceback.format_exc())
                
                task_state.status = TaskStatus.FAILED
                task_state.error = f"Completion handler error: {str(e)}"
                self.execution_state.failed_tasks.add(task_id)
                
                if self.fail_fast:
                    self.stop_execution("completion_error")
        
        # Notify completion and process ready tasks
        self.result_queue.put(task_id)
        self._process_ready_tasks()
    
    def _process_ready_tasks(self):
        """Process tasks that are now ready to execute"""
        if self.execution_state.execution_stopped:
            return
            
        ready_tasks = []
        with self._lock:
            for task_state in self.execution_state.current_tasks.values():
                if (task_state.status == TaskStatus.QUEUED and 
                    self._is_task_ready(task_state)):
                    ready_tasks.append(task_state.task_id)
        
        # Execute ready tasks
        for task_id in ready_tasks:
            logger.debug(f"Processing ready task: {task_id}")
            self._execute_task(task_id)
        
        if ready_tasks:
            self.event_bus.emit(
                EventType.READY_TASKS_PROCESSED,
                ready_tasks_count=len(ready_tasks)
            )
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> List[int]:
        """Wait for task completions with improved timeout handling"""
        completed = []
        start_time = time.time()
        
        if self._execution_start_time is None:
            self._execution_start_time = datetime.now()
        
        while not self.all_tasks_finished():
            try:
                remaining_time = None
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_time = max(0, timeout - elapsed)
                    if remaining_time <= 0:
                        logger.warning("Timeout waiting for task completion")
                        break
                
                task_id = self.result_queue.get(timeout=remaining_time)
                completed.append(task_id)
                
                # Log progress
                with self._lock:
                    finished = len(self.execution_state.completed_tasks) + len(self.execution_state.failed_tasks) + len(self.execution_state.cancelled_tasks) + len(self.execution_state.skipped_tasks)
                    total = self.execution_state.total_tasks
                    if total > 0:
                        progress = finished / total * 100
                        logger.debug(f"Execution progress: {progress:.1f}% ({finished}/{total})")
                    
            except queue.Empty:
                if timeout:
                    logger.warning("Timeout reached while waiting for completions")
                break
        
        return completed
    
    def all_tasks_finished(self) -> bool:
        """Check if all tasks are finished"""
        if self.execution_state.execution_stopped:
            return True
            
        with self._lock:
            for task_state in self.execution_state.current_tasks.values():
                if task_state.status in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING]:
                    return False
        return True
    
    def stop_execution(self, reason: str = "user_request"):
        """Stop all task execution immediately with better cleanup"""
        logger.warning(f"Stopping all task execution: {reason}")
        
        cancelled_tasks = []
        running_tasks = []
        
        with self._lock:
            self.execution_state.execution_stopped = True
            
            # Cancel pending and queued tasks, collect running tasks
            for task_state in self.execution_state.current_tasks.values():
                if task_state.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                    task_state.status = TaskStatus.CANCELLED
                    task_state.end_time = datetime.now()
                    self.execution_state.cancelled_tasks.add(task_state.task_id)
                    cancelled_tasks.append(task_state.task_id)
                    
                elif task_state.status == TaskStatus.RUNNING:
                    running_tasks.append((task_state.task_id, task_state.future))
        
        # Emit cancellation events
        for task_id in cancelled_tasks:
            task_state = self.execution_state.current_tasks[task_id]
            self.event_bus.emit(
                EventType.TASK_CANCELLED,
                task_id=task_id,
                task_name=task_state.name,
                reason=reason
            )
            self.result_queue.put(task_id)
        
        # Cancel running tasks (best effort)
        for task_id, future in running_tasks:
            if future and not future.done():
                future.cancel()
                logger.debug(f"Attempted to cancel running task {task_id}")
        
        self.event_bus.emit(EventType.EXECUTION_STOPPED, reason=reason)
    
    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed"""
        return len(self.execution_state.failed_tasks) > 0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a comprehensive execution summary"""
        with self._lock:
            state = self.execution_state
            duration_ms = None
            
            if self._execution_start_time:
                duration_ms = (datetime.now() - self._execution_start_time).total_seconds() * 1000
            
            return {
                "total_tasks": state.total_tasks,
                "completed": len(state.completed_tasks),
                "failed": len(state.failed_tasks),
                "cancelled": len(state.cancelled_tasks),
                "skipped": len(state.skipped_tasks),
                "running": sum(1 for t in state.current_tasks.values() if t.status == TaskStatus.RUNNING),
                "queued": sum(1 for t in state.current_tasks.values() if t.status == TaskStatus.QUEUED),
                "execution_stopped": state.execution_stopped,
                "duration_ms": duration_ms,
                "success_rate": len(state.completed_tasks) / max(1, state.total_tasks) * 100
            }
    
    def reset_for_replan(self):
        """Reset state for replanning with better cleanup"""
        logger.info("Resetting executor for replanning")
        
        with self._lock:
            # Cancel any running tasks first
            for task_state in self.execution_state.current_tasks.values():
                if task_state.future and not task_state.future.done():
                    task_state.future.cancel()
            
            # Reset state
            self.execution_state.execution_stopped = False
            self.execution_state.current_tasks.clear()
            self.execution_state.completed_tasks.clear()
            self.execution_state.failed_tasks.clear()
            self.execution_state.cancelled_tasks.clear()
            self.execution_state.skipped_tasks.clear()
            self.execution_state.total_tasks = 0
            self.execution_state.observations.clear()
            
            self._task_registry.clear()
            self._execution_start_time = None
            
        # Clear result queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    @contextmanager
    def execution_context(self):
        """Context manager for safe execution"""
        try:
            self._execution_start_time = datetime.now()
            yield self
        except Exception as e:
            logger.error(f"Execution context error: {e}")
            self.stop_execution("context_error")
            raise
        finally:
            # Ensure cleanup
            pass
    
    def shutdown(self):
        """Shutdown executor with proper cleanup"""
        logger.info("Shutting down task executor")
        self._shutdown = True
        
        # Stop execution first
        if not self.execution_state.execution_stopped:
            self.stop_execution("shutdown")
        
        # Wait a bit for running tasks to complete
        try:
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during executor shutdown: {e}")
            # Force shutdown
            self.executor.shutdown(wait=False)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()