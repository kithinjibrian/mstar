from app.settings import *  # env keys
from litellm import completion
import json
import re
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
import time

class ThoughtStatus(Enum):
    ACTIVE = "active"
    EVALUATED = "evaluated" 
    SELECTED = "selected"
    PRUNED = "pruned"

class SearchStrategy(Enum):
    BREADTH_FIRST = "breadth_first" 
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"

@dataclass
class SearchPath:
    """Represents a path through the thought tree"""
    thoughts: List['Thought']
    current_step: int
    parent: Optional['Thought']
    total_score: float = 0.0
    avg_score: float = 0.0
    
    def __post_init__(self):
        if self.thoughts:
            self.total_score = sum(t.score for t in self.thoughts)
            self.avg_score = self.total_score / len(self.thoughts)
    
    def __lt__(self, other):
        # For heap operations - higher scores have higher priority
        return self.avg_score > other.avg_score

@dataclass
class Thought:
    content: str
    step: int
    score: float = 0.0
    reasoning: str = ""
    parent_id: Optional[str] = None
    thought_id: str = field(default_factory=lambda: f"thought_{int(time.time() * 1000000) % 1000000}")
    status: ThoughtStatus = ThoughtStatus.ACTIVE
    metadata: Dict = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "id": self.thought_id,
            "content": self.content,
            "step": self.step,
            "score": self.score,
            "reasoning": self.reasoning,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "metadata": self.metadata,
            "children": self.children
        }

class ToTVisualizer:
    """Handles real-time visualization of the thought tree"""
    
    @staticmethod
    def print_tree_status(tot_instance):
        """Print current tree structure"""
        print("\n" + "🌳 THOUGHT TREE STATUS " + "="*30)
        
        # Group thoughts by step
        steps_thoughts = {}
        for thought in tot_instance.thoughts.values():
            if thought.step not in steps_thoughts:
                steps_thoughts[thought.step] = []
            steps_thoughts[thought.step].append(thought)
        
        # Display each step
        for step_idx in sorted(steps_thoughts.keys()):
            step_desc = tot_instance.steps[step_idx] if step_idx < len(tot_instance.steps) else "Unknown step"
            print(f"\n📍 Step {step_idx + 1}: {step_desc[:60]}...")
            
            thoughts = sorted(steps_thoughts[step_idx], key=lambda t: t.score, reverse=True)
            for thought in thoughts:
                status_emoji = {
                    ThoughtStatus.ACTIVE: "🔵",
                    ThoughtStatus.EVALUATED: "🟡", 
                    ThoughtStatus.SELECTED: "🟢",
                    ThoughtStatus.PRUNED: "🔴"
                }
                
                parent_info = f" (parent: {thought.parent_id[-6:]})" if thought.parent_id else ""
                score_info = f" [{thought.score:.1f}/10]" if thought.score > 0 else ""
                
                print(f"  {status_emoji[thought.status]} {thought.thought_id[-6:]}: {thought.content[:80]}...{score_info}{parent_info}")
        
        print(f"\n📊 Total thoughts: {len(tot_instance.thoughts)}")
        print("="*50)
    
    @staticmethod
    def print_search_strategy_info(strategy: SearchStrategy, beam_width: int = None):
        """Print information about the current search strategy"""
        strategy_info = {
            SearchStrategy.BREADTH_FIRST: "🌊 BREADTH-FIRST SEARCH - Explores all paths equally",
            SearchStrategy.GREEDY: "⚡ GREEDY SEARCH - Always picks the best thought at each step",
            SearchStrategy.BEAM_SEARCH: f"🔦 BEAM SEARCH (width={beam_width}) - Keeps top {beam_width} paths at each step"
        }
        
        print(f"\n{strategy_info.get(strategy, 'Unknown strategy')}")
        print("="*60)
    
    @staticmethod
    def print_ascii_tree(tot_instance):
        """Print ASCII tree representation of thought hierarchy"""
        print("\n" + "🌲 ASCII TREE STRUCTURE " + "="*25)
        
        # Find root thoughts (no parent)
        root_thoughts = [t for t in tot_instance.thoughts.values() if not t.parent_id]
        
        if not root_thoughts:
            print("No thoughts in tree yet.")
            return
        
        # Sort roots by step and score
        root_thoughts.sort(key=lambda t: (t.step, -t.score))
        
        for i, root in enumerate(root_thoughts):
            is_last_root = (i == len(root_thoughts) - 1)
            ToTVisualizer._print_thought_branch(tot_instance, root, "", is_last_root)
        
        print("="*50)
    
    @staticmethod
    def _print_thought_branch(tot_instance, thought, prefix, is_last):
        """Recursively print a thought and its children with ASCII tree formatting"""
        # Status symbols
        status_symbols = {
            ThoughtStatus.ACTIVE: "○",
            ThoughtStatus.EVALUATED: "◑", 
            ThoughtStatus.SELECTED: "●",
            ThoughtStatus.PRUNED: "✗"
        }
        
        # Tree connectors
        connector = "└── " if is_last else "├── "
        
        # Thought info
        status = status_symbols.get(thought.status, "?")
        score_info = f"[{thought.score:.1f}]" if thought.score > 0 else "[--]"
        step_info = f"S{thought.step + 1}"
        id_short = thought.thought_id[-4:]
        content_short = thought.content[:50].replace('\n', ' ')
        
        # Print the thought
        print(f"{prefix}{connector}{status} {step_info}:{id_short} {score_info} {content_short}...")
        
        # Prepare prefix for children
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Get and sort children
        children = []
        for child_id in thought.children:
            if child_id in tot_instance.thoughts:
                children.append(tot_instance.thoughts[child_id])
        
        # Sort children by score (best first)
        children.sort(key=lambda t: -t.score)
        
        # Print children
        for j, child in enumerate(children):
            is_last_child = (j == len(children) - 1)
            ToTVisualizer._print_thought_branch(tot_instance, child, child_prefix, is_last_child)
    
    @staticmethod
    def print_solution_path_tree(solution_path):
        """Print ASCII tree showing only the solution path"""
        if not solution_path:
            print("No solution path found.")
            return
            
        print("\n" + "🏆 SOLUTION PATH TREE " + "="*25)
        
        for i, thought in enumerate(solution_path):
            is_last = (i == len(solution_path) - 1)
            connector = "└── " if is_last else "├── "
            prefix = "    " * i
            
            step_info = f"Step {thought.step + 1}"
            score_info = f"[{thought.score:.1f}/10]"
            content_short = thought.content[:60].replace('\n', ' ')
            
            print(f"{prefix}{connector}● {step_info} {score_info} {content_short}...")
            
            if not is_last:
                print(f"{prefix}│")
        
        avg_score = sum(t.score for t in solution_path) / len(solution_path)
        print(f"\n🎯 Solution Path Score: {avg_score:.1f}/10")
        print("="*50)
    
    @staticmethod
    def print_beam_status(active_paths: List[SearchPath], beam_width: int):
        """Print current beam search status"""
        print(f"\n📡 BEAM STATUS (keeping top {beam_width} paths)")
        print("-" * 40)
        
        for i, path in enumerate(active_paths[:beam_width]):
            print(f"Path {i+1}: avg_score={path.avg_score:.1f}, steps={len(path.thoughts)}, current_step={path.current_step}")
            if path.thoughts:
                latest = path.thoughts[-1]
                print(f"  Latest: {latest.content[:50]}...")
        print("-" * 40)

class JSONParser:
    """Fault-tolerant JSON parser that handles markdown and other formatting issues"""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict|List]:
        """Extract JSON from text that might contain markdown or other formatting"""
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try direct JSON parsing first
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Look for JSON-like structures
        json_patterns = [
            r'\[.*\]',  # Array
            r'\{.*\}',  # Object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # Try to extract from lines that look like JSON
        lines = text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('[') or line.startswith('{'):
                in_json = True
            if in_json:
                json_lines.append(line)
            if line.endswith(']') or line.endswith('}'):
                break
        
        if json_lines:
            try:
                return json.loads('\n'.join(json_lines))
            except:
                pass
        
        return None
    
    @staticmethod
    def parse_list_fallback(text: str, expected_count: int = 3) -> List[str]:
        """Fallback to extract list items from text"""
        items = []
        
        # Look for numbered lists
        numbered_pattern = r'^\d+\.\s*(.+)$'
        for line in text.split('\n'):
            match = re.match(numbered_pattern, line.strip())
            if match:
                items.append(match.group(1))
        
        # Look for bullet points
        if not items:
            bullet_pattern = r'^[-*•]\s*(.+)$'
            for line in text.split('\n'):
                match = re.match(bullet_pattern, line.strip())
                if match:
                    items.append(match.group(1))
        
        # Look for quoted items
        if not items:
            quote_pattern = r'"([^"]+)"'
            items = re.findall(quote_pattern, text)
        
        # Return up to expected count
        return items[:expected_count] if items else [f"Item {i+1}" for i in range(expected_count)]

class ToT:
    def __init__(self, model="deepseek/deepseek-chat", max_thoughts_per_step=3, max_depth=5, score_threshold=6.0, 
                 enable_streaming=True, show_tree_updates=True, search_strategy=SearchStrategy.GREEDY, 
                 beam_width=3):
        self.model = model
        self.max_thoughts_per_step = max_thoughts_per_step
        self.max_depth = max_depth
        self.score_threshold = score_threshold
        self.enable_streaming = enable_streaming
        self.show_tree_updates = show_tree_updates
        self.search_strategy = search_strategy
        self.beam_width = beam_width
        
        # Store all thoughts in a flat structure for easy access
        self.thoughts: Dict[str, Thought] = {}
        self.steps: List[str] = []  # Decomposed steps
        self.problem: str = ""
        self.current_step = 0
        
        self.visualizer = ToTVisualizer()
        self.parser = JSONParser()
        
    def _stream_response(self, **completion_kwargs) -> Tuple[str, Generator]:
        """Handle streaming or non-streaming response"""
        if self.enable_streaming:
            completion_kwargs['stream'] = True
            response = completion(**completion_kwargs)
            
            full_content = ""
            print("💭 ", end="", flush=True)
            
            for chunk in response:
                if chunk.choices[0].delta.get("reasoning_content"):
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                    print(reasoning_content, end="", flush=True)

                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_content += content
            
            print()  # New line after streaming
            return full_content, None
        else:
            response = completion(**completion_kwargs)
            content = response.choices[0].message.content
            print(f"💭 {content}")
            return content, None
    
    def solve(self, problem: str) -> Dict:
        """Main method to solve a problem using Tree of Thoughts"""
        self.problem = problem
        print(f"🎯 Starting ToT for problem: {problem[:100]}...")
        
        # Show search strategy info
        self.visualizer.print_search_strategy_info(self.search_strategy, self.beam_width)
        
        # Step 1: Decompose the problem
        print("\n🔍 DECOMPOSITION PHASE")
        self.steps = self.decompose()
        print(f"📋 Decomposed into {len(self.steps)} steps:")
        for i, step in enumerate(self.steps):
            print(f"  {i+1}. {step}")
        
        # Step 2: Generate and search through thoughts
        print(f"\n🧠 THOUGHT GENERATION & SEARCH PHASE")
        solution_path = self.search()
        
        # Step 3: Compile final solution
        print(f"\n📝 SOLUTION COMPILATION PHASE")
        final_solution = self._compile_solution(solution_path)
        
        return {
            "problem": problem,
            "steps": self.steps,
            "solution_path": [t.to_dict() for t in solution_path],
            "final_solution": final_solution,
            "total_thoughts_generated": len(self.thoughts),
            "search_strategy": self.search_strategy.value,
            "beam_width": self.beam_width if self.search_strategy == SearchStrategy.BEAM_SEARCH else None,
            "tree_structure": self._get_tree_structure()
        }
    
    def decompose(self) -> List[str]:
        """Decompose a problem into logical steps with streaming"""
        prompt = f"""
        Problem: {self.problem}
        
        Break this problem down into clear and logical steps that would lead to a solution.
        Each step should be:
        - Specific and actionable
        - Build on previous steps
        - Move closer to solving the original problem
        
        Return your response as a JSON list of strings, like:
        ["Step 1 description", "Step 2 description", ...]
        
        Focus on the logical flow rather than implementation details.
        """
        
        content, _ = self._stream_response(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # Parse with fault tolerance
        parsed_steps = self.parser.extract_json_from_text(content)
        
        if parsed_steps and isinstance(parsed_steps, list):
            return parsed_steps
        else:
            # Fallback parsing
            print("⚠️ JSON parsing failed, using fallback method")
            return self.parser.parse_list_fallback(content, 4)
    
    def generate(self, step_index: int, parent_thought: Optional[Thought] = None) -> List[Thought]:
        """Generate N thoughts for a specific step with streaming"""
        step_description = self.steps[step_index]
        
        context = f"Problem: {self.problem}\n"
        context += f"Current Step ({step_index + 1}/{len(self.steps)}): {step_description}\n"
        
        if parent_thought:
            context += f"Building on previous thought: {parent_thought.content}\n"
        
        prompt = f"""
        {context}
        
        Generate {self.max_thoughts_per_step} different approaches for this step.
        Each approach should be:
        - Distinct from the others
        - Practical and feasible
        - Clearly explained
        
        Return your response as a JSON array of objects with this format:
        [
            {{"approach": "Detailed description of approach 1", "reasoning": "Why this approach makes sense"}},
            {{"approach": "Detailed description of approach 2", "reasoning": "Why this approach makes sense"}},
            ...
        ]
        """
        
        print(f"\n🔄 Generating thoughts for step {step_index + 1}...")
        content, _ = self._stream_response(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        # Parse with fault tolerance
        approaches = self.parser.extract_json_from_text(content)
        
        if not approaches or not isinstance(approaches, list):
            print("⚠️ JSON parsing failed, using fallback approach generation")
            approaches = [
                {"approach": f"Approach {i+1}: {step_description}", "reasoning": f"Generated approach {i+1}"}
                for i in range(self.max_thoughts_per_step)
            ]
        
        thoughts = []
        for i, approach_data in enumerate(approaches[:self.max_thoughts_per_step]):
            if isinstance(approach_data, dict):
                approach_text = approach_data.get("approach", f"Approach {i+1}")
                reasoning_text = approach_data.get("reasoning", "Auto-generated reasoning")
            else:
                approach_text = str(approach_data)
                reasoning_text = "Parsed from text"
            
            thought = Thought(
                content=approach_text,
                step=step_index,
                reasoning=reasoning_text,
                parent_id=parent_thought.thought_id if parent_thought else None
            )
            
            # Update parent's children list
            if parent_thought:
                parent_thought.children.append(thought.thought_id)
            
            thoughts.append(thought)
            self.thoughts[thought.thought_id] = thought
            
        print(f"✅ Generated {len(thoughts)} thoughts")
        
        # Show tree update if enabled
        if self.show_tree_updates:
            self.visualizer.print_tree_status(self)
            
        return thoughts
    
    def eval(self, thoughts: List[Thought]) -> List[Thought]:
        """Evaluate and score thoughts with streaming"""
        if not thoughts:
            return thoughts
            
        # Prepare thoughts for evaluation
        thoughts_text = ""
        for i, thought in enumerate(thoughts):
            thoughts_text += f"Thought {i+1}: {thought.content}\n"
            thoughts_text += f"Reasoning: {thought.reasoning}\n\n"
        
        prompt = f"""
        Problem: {self.problem}
        Current Step: {self.steps[thoughts[0].step]}
        
        Evaluate these {len(thoughts)} approaches:
        
        {thoughts_text}
        
        Rate each thought on a scale of 1-10 considering:
        - How well it addresses the current step
        - Feasibility and practicality
        - How well it sets up future steps
        - Creativity and effectiveness
        
        Return your response as a JSON array of objects:
        [
            {{"thought_number": 1, "score": 8.5, "evaluation": "Brief explanation of score"}},
            {{"thought_number": 2, "score": 6.2, "evaluation": "Brief explanation of score"}},
            ...
        ]
        """
        
        print(f"📊 Evaluating {len(thoughts)} thoughts...")
        content, _ = self._stream_response(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        # Parse evaluations with fault tolerance
        evaluations = self.parser.extract_json_from_text(content)
        
        if evaluations and isinstance(evaluations, list):
            for eval_data in evaluations:
                if isinstance(eval_data, dict):
                    thought_idx = eval_data.get("thought_number", 1) - 1
                    if 0 <= thought_idx < len(thoughts):
                        score = eval_data.get("score", 5.0)
                        # Handle string scores
                        if isinstance(score, str):
                            try:
                                score = float(score)
                            except:
                                score = 5.0
                        
                        thoughts[thought_idx].score = score
                        thoughts[thought_idx].metadata["evaluation"] = eval_data.get("evaluation", "")
                        thoughts[thought_idx].status = ThoughtStatus.EVALUATED
        else:
            print("⚠️ Evaluation parsing failed, using fallback scoring")
            # Fallback: assign scores based on content length and keywords
            for i, thought in enumerate(thoughts):
                # Simple heuristic scoring
                score = 5.0
                if "innovative" in thought.content.lower() or "creative" in thought.content.lower():
                    score += 1
                if len(thought.content) > 100:
                    score += 0.5
                if "step" in thought.content.lower():
                    score += 0.5
                
                thought.score = min(score, 10.0)
                thought.status = ThoughtStatus.EVALUATED
        
        sorted_thoughts = sorted(thoughts, key=lambda t: t.score, reverse=True)
        
        print(f"📈 Evaluation complete. Scores: {[f'{t.score:.1f}' for t in sorted_thoughts]}")
        
        return sorted_thoughts
    
    def search(self) -> List[Thought]:
        """Search for the best solution path using the specified strategy"""
        if self.search_strategy == SearchStrategy.GREEDY:
            return self._greedy_search()
        elif self.search_strategy == SearchStrategy.BEAM_SEARCH:
            return self._beam_search()
        else:
            return self._breadth_first_search()
    
    def _breadth_first_search(self) -> List[Thought]:
        """Original breadth-first search implementation"""
        print("🔍 Starting breadth-first search through thought space...")
        
        current_paths = [SearchPath(thoughts=[], current_step=0, parent=None)]
        best_complete_path = None
        best_score = 0
        
        while current_paths and not best_complete_path:
            next_paths = []
            
            for path in current_paths:
                step_idx = path.current_step
                
                # Check if we've completed all steps
                if step_idx >= len(self.steps):
                    if path.avg_score > best_score:
                        best_score = path.avg_score
                        best_complete_path = path.thoughts
                    continue
                
                print(f"\n🌟 Processing step {step_idx + 1}: {self.steps[step_idx][:60]}...")
                
                # Generate thoughts for current step
                thoughts = self.generate(step_idx, path.parent)
                
                # Evaluate thoughts
                evaluated_thoughts = self.eval(thoughts)
                
                # Keep top thoughts that meet threshold
                good_thoughts = [t for t in evaluated_thoughts if t.score >= self.score_threshold]
                if not good_thoughts and evaluated_thoughts:
                    print(f"⚠️ No thoughts met threshold {self.score_threshold}, keeping best one")
                    good_thoughts = [evaluated_thoughts[0]]
                
                # Create new paths for each good thought
                for thought in good_thoughts[:2]:  # Limit branching factor
                    thought.status = ThoughtStatus.SELECTED
                    new_path = SearchPath(
                        thoughts=path.thoughts + [thought],
                        current_step=step_idx + 1,
                        parent=thought
                    )
                    next_paths.append(new_path)
                    print(f"✅ Selected thought {thought.thought_id[-6:]} (score: {thought.score:.1f})")
            
            current_paths = next_paths
            
            # Prevent infinite loops
            if len(self.thoughts) > 50:
                print("⚠️ Thought limit reached, selecting best current path")
                break
        
        if not best_complete_path and current_paths:
            best_partial = max(current_paths, key=lambda p: p.avg_score)
            best_complete_path = best_partial.thoughts
        
        print(f"\n🎉 Found solution path with {len(best_complete_path or [])} thoughts!")
        return best_complete_path or []
    
    def _greedy_search(self) -> List[Thought]:
        """Greedy search - always picks the best thought at each step"""
        print("⚡ Starting greedy search through thought space...")
        
        solution_path = []
        current_parent = None
        
        for step_idx in range(len(self.steps)):
            print(f"\n🎯 Greedy step {step_idx + 1}: {self.steps[step_idx][:60]}...")
            
            # Generate thoughts for current step
            thoughts = self.generate(step_idx, current_parent)
            
            # Evaluate thoughts
            evaluated_thoughts = self.eval(thoughts)
            
            if not evaluated_thoughts:
                print("⚠️ No thoughts generated for this step")
                break
            
            # Greedy choice: always pick the best scoring thought
            best_thought = evaluated_thoughts[0]  # Already sorted by score descending
            best_thought.status = ThoughtStatus.SELECTED
            
            # Mark others as pruned
            for thought in evaluated_thoughts[1:]:
                thought.status = ThoughtStatus.PRUNED
            
            solution_path.append(best_thought)
            current_parent = best_thought
            
            print(f"✅ Greedy choice: {best_thought.thought_id[-6:]} (score: {best_thought.score:.1f})")
            
            # Show tree update if enabled
            if self.show_tree_updates:
                self.visualizer.print_ascii_tree(self)
        
        print(f"\n🎉 Greedy search complete with {len(solution_path)} thoughts!")
        return solution_path
    
    def _beam_search(self) -> List[Thought]:
        """Beam search - keeps top K paths at each step"""
        print(f"🔦 Starting beam search (width={self.beam_width}) through thought space...")
        
        # Priority queue to maintain top beam_width paths
        active_paths = [SearchPath(thoughts=[], current_step=0, parent=None)]
        
        for step_idx in range(len(self.steps)):
            print(f"\n🔦 Beam step {step_idx + 1}: {self.steps[step_idx][:60]}...")
            
            if self.show_tree_updates:
                self.visualizer.print_beam_status(active_paths, self.beam_width)
            
            all_new_paths = []
            
            # Expand each path in the current beam
            for path in active_paths:
                if path.current_step != step_idx:
                    continue
                    
                # Generate thoughts for current step
                thoughts = self.generate(step_idx, path.parent)
                
                # Evaluate thoughts
                evaluated_thoughts = self.eval(thoughts)
                
                # Create new paths for each thought
                for thought in evaluated_thoughts:
                    new_path = SearchPath(
                        thoughts=path.thoughts + [thought],
                        current_step=step_idx + 1,
                        parent=thought
                    )
                    all_new_paths.append(new_path)
            
            # Keep only top beam_width paths
            all_new_paths.sort(key=lambda p: p.avg_score, reverse=True)
            active_paths = all_new_paths[:self.beam_width]
            
            # Update thought statuses
            selected_thoughts = set()
            for path in active_paths:
                if path.thoughts:
                    selected_thoughts.add(path.thoughts[-1].thought_id)
            
            # Mark thoughts as selected or pruned
            for thought in self.thoughts.values():
                if thought.step == step_idx and thought.status == ThoughtStatus.EVALUATED:
                    if thought.thought_id in selected_thoughts:
                        thought.status = ThoughtStatus.SELECTED
                    else:
                        thought.status = ThoughtStatus.PRUNED
            
            print(f"✅ Kept top {len(active_paths)} paths for beam")
            
            # Show tree update if enabled
            if self.show_tree_updates:
                self.visualizer.print_ascii_tree(self)
        
        # Return the best complete path
        if active_paths:
            best_path = max(active_paths, key=lambda p: p.avg_score)
            print(f"\n🎉 Beam search complete! Best path score: {best_path.avg_score:.1f}")
            return best_path.thoughts
        
        return []
    
    def _compile_solution(self, solution_path: List[Thought]) -> str:
        """Compile the solution path into a final answer"""
        if not solution_path:
            return "No solution found."
        
        solution_text = f"{self.problem}\n"
        if self.search_strategy == SearchStrategy.BEAM_SEARCH:
            solution_text += f"Beam Width: {self.beam_width}\n"
        solution_text += "\n"
        
        for i, thought in enumerate(solution_path):
            step_desc = self.steps[thought.step] if thought.step < len(self.steps) else "Final step"
            solution_text += f"Step {thought.step + 1}: {step_desc}\n"
            solution_text += f"Approach: {thought.content}\n"
            solution_text += f"Reasoning: {thought.reasoning}\n"
            solution_text += f"Score: {thought.score:.1f}/10\n\n"
        
        return solution_text
    
    def _get_tree_structure(self) -> Dict:
        """Get a hierarchical representation of the thought tree"""
        tree = {}
        
        for thought in self.thoughts.values():
            if not thought.parent_id:  # Root thoughts
                tree[thought.thought_id] = self._build_subtree(thought)
        
        return tree
    
    def _build_subtree(self, thought: Thought) -> Dict:
        """Recursively build subtree for a thought"""
        subtree = thought.to_dict()
        subtree["children"] = {}
        
        for child_id in thought.children:
            if child_id in self.thoughts:
                child_thought = self.thoughts[child_id]
                subtree["children"][child_id] = self._build_subtree(child_thought)
        
        return subtree

def compare_search_strategies(problem: str):
    """Compare different search strategies on the same problem"""
    print("🔬 COMPARING SEARCH STRATEGIES")
    print("="*60)
    
    strategies = [
        (SearchStrategy.GREEDY, None),
        (SearchStrategy.BEAM_SEARCH, 2),
        (SearchStrategy.BEAM_SEARCH, 3),
        (SearchStrategy.BREADTH_FIRST, None)
    ]
    
    results = {}
    
    for strategy, beam_width in strategies:
        strategy_name = strategy.value + (f"_w{beam_width}" if beam_width else "")
        print(f"\n🧪 Testing {strategy_name}...")
        
        tot = ToT(
            max_thoughts_per_step=3,
            max_depth=4,
            score_threshold=6.0,
            enable_streaming=False,  # Disable for comparison
            show_tree_updates=False,  # Disable for cleaner output
            search_strategy=strategy,
            beam_width=beam_width or 3
        )
        
        result = tot.solve(problem)
        
        results[strategy_name] = {
            "avg_score": sum(t["score"] for t in result["solution_path"]) / len(result["solution_path"]) if result["solution_path"] else 0,
            "total_thoughts": result["total_thoughts_generated"],
            "path_length": len(result["solution_path"]),
            "strategy": strategy.value,
            "beam_width": beam_width
        }
        
        print(f"✅ {strategy_name}: avg_score={results[strategy_name]['avg_score']:.1f}, thoughts={results[strategy_name]['total_thoughts']}")
    
    # Print comparison table
    print(f"\n📊 STRATEGY COMPARISON RESULTS")
    print("="*80)
    print(f"{'Strategy':<20} {'Avg Score':<12} {'Total Thoughts':<15} {'Path Length':<12}")
    print("-"*80)
    
    for name, data in results.items():
        print(f"{name:<20} {data['avg_score']:<12.1f} {data['total_thoughts']:<15} {data['path_length']:<12}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['avg_score'])
    print(f"\n🏆 Best performing strategy: {best_strategy[0]} (avg score: {best_strategy[1]['avg_score']:.1f})")
    
    return results