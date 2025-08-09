from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import threading
from typing import Any, Callable, Dict, List


logging.basicConfig(level=None)
logger = logging.getLogger(__name__)

class EventType(Enum):
    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    
    # Planning events
    PLANNING_STARTED = "planning_started"
    PLANNING_STREAM_CHUNK = "planning_stream_chunk"
    PLANNING_COMPLETED = "planning_completed"
    PLANNING_FAILED = "planning_failed"
    
    # Task events
    TASK_PARSED = "task_parsed"
    TASK_QUEUED = "task_queued"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    
    # Execution control events
    EXECUTION_STOPPED = "execution_stopped"
    READY_TASKS_PROCESSED = "ready_tasks_processed"
    
    # Joining events
    JOINING_STARTED = "joining_started"
    JOINING_THINKING = "joining_thinking"
    JOINING_COMPLETED = "joining_completed"
    JOINING_FAILED = "joining_failed"
    
    # AI Response events
    AI_STREAM_CHUNK = "ai_stream_chunk"
    AI_RESPONSE_COMPLETED = "ai_response_completed"
    
    # Replanning events
    REPLAN_TRIGGERED = "replan_triggered"
    REPLAN_CONTEXT_CREATED = "replan_context_created"


@dataclass
class ObservabilityEvent:
    """Event emitted during agent execution for UI observability"""
    event_type: EventType
    timestamp: datetime
    agent_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "data": self.data
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)

class EventBus:
    """Central event bus for observability"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.callbacks: Dict[EventType, List[Callable]] = {}
        self.global_callbacks: List[Callable] = []
        self.event_history: List[ObservabilityEvent] = []
        self._lock = threading.Lock()
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to specific event type"""
        with self._lock:
            if event_type not in self.callbacks:
                self.callbacks[event_type] = []
            self.callbacks[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable):
        """Subscribe to all events"""
        with self._lock:
            self.global_callbacks.append(callback)
    
    def emit(self, event_type: EventType, **data):
        """Emit an event"""
        event = ObservabilityEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            data=data
        )
        
        with self._lock:
            self.event_history.append(event)
            
            # Call specific callbacks
            for callback in self.callbacks.get(event_type, []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
            
            # Call global callbacks  
            for callback in self.global_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Global callback error: {e}")
    
    def get_history(self) -> List[ObservabilityEvent]:
        with self._lock:
            return self.event_history.copy()
    
    def clear_history(self):
        with self._lock:
            self.event_history.clear()