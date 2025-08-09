from typing import Dict, List, Callable, Any, Optional, Union
from collections import defaultdict, deque

START = "__start__"
END = "__end__"

class Graph:
    """A simple LangGraph clone with basic graph operations and execution."""
    
    def __init__(self):
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)
        self.conditional_edges: Dict[str, Dict[str, Any]] = {}
        self.entry_point: Optional[str] = START
    
    def add_node(self, name: str, func: Callable) -> None:
        """Add a node to the graph.
        
        Args:
            name: Unique identifier for the node
            func: Function to execute when this node is reached
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        self.nodes[name] = func
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a direct edge between two nodes.
        
        Args:
            from_node: Source node name (can be '__start__' for entry)
            to_node: Destination node name (can be '__end__' for termination)
        """
        if from_node != START and from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' does not exist")
        if to_node != END and to_node not in self.nodes:
            raise ValueError(f"Destination node '{to_node}' does not exist")
        
        # Remove any existing conditional edge from this node
        if from_node in self.conditional_edges:
            del self.conditional_edges[from_node]
        
        self.edges[from_node].append(to_node)
    
    def add_conditional_edge(self, from_node: str, condition_func: Callable, 
                           edge_map: Dict[Any, str]) -> None:
        """Add a conditional edge that chooses destination based on condition result.
        
        Args:
            from_node: Source node name (can be '__start__' for entry)
            condition_func: Function that returns a key to determine next node
            edge_map: Mapping from condition result to destination node name (can include '__end__')
        """
        if from_node != START and from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' does not exist")
        
        # Validate all destination nodes exist (except __end__)
        for dest_node in edge_map.values():
            if dest_node != END and dest_node not in self.nodes:
                raise ValueError(f"Destination node '{dest_node}' does not exist")
        
        # Remove any existing direct edges from this node
        if from_node in self.edges:
            self.edges[from_node] = []
        
        self.conditional_edges[from_node] = {
            'condition': condition_func,
            'edge_map': edge_map
        }
    
    def set_entry_point(self, node_name: str) -> None:
        """Set the entry point for graph execution.
        
        Args:
            node_name: Name of the node to start execution from, or '__start__' for automatic start
        """
        if node_name != START and node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' does not exist")
        self.entry_point = node_name
    
    def execute(self, initial_state: Any = None) -> Any:
        """Execute the graph starting from the entry point.
        
        Args:
            initial_state: Initial state to pass to the first node
            
        Returns:
            Final state after graph execution
        """
        if not self.entry_point:
            raise ValueError("No entry point set. Use set_entry_point() first.")
        
        # Handle special __start__ entry point
        if self.entry_point == START:
            current_node = self._get_next_node(START, initial_state)
        else:
            current_node = self.entry_point
            
        state = initial_state
        visited = []
        
        while current_node and current_node != END:
            visited.append(current_node)
            
            # Execute current node
            node_func = self.nodes[current_node]
            try:
                state = node_func(state)
            except Exception as e:
                raise RuntimeError(f"Error executing node '{current_node}': {e}")
            
            # Determine next node
            next_node = self._get_next_node(current_node, state)
            
            # Prevent infinite loops (simple check)
            if len(visited) > 100:
                raise RuntimeError("Possible infinite loop detected (>100 nodes executed)")
            
            current_node = next_node
        
        return state
    
    def _get_next_node(self, current_node: str, state: Any) -> Optional[str]:
        """Determine the next node to execute based on edges and conditions."""
        
        # Check for conditional edges first
        if current_node in self.conditional_edges:
            cond_info = self.conditional_edges[current_node]
            condition_func = cond_info['condition']
            edge_map = cond_info['edge_map']
            
            try:
                condition_result = condition_func(state)
                if condition_result in edge_map:
                    return edge_map[condition_result]
                else:
                    raise ValueError(f"Condition result '{condition_result}' not found in edge map")
            except Exception as e:
                raise RuntimeError(f"Error evaluating condition for node '{current_node}': {e}")
        
        # Check for direct edges
        if current_node in self.edges and self.edges[current_node]:
            # If multiple edges, take the first one (for simplicity)
            return self.edges[current_node][0]
        
        # No outgoing edges - end execution
        return None
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the current graph structure."""
        return {
            'nodes': list(self.nodes.keys()),
            'edges': dict(self.edges),
            'conditional_edges': {k: {'edge_map': v['edge_map']} 
                                for k, v in self.conditional_edges.items()},
            'entry_point': self.entry_point
        }