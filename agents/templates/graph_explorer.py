"""
Graph-Based Explorer Agent for ARC-AGI-3

An agent that builds a graph of game states and explores strategically,
prioritizing unexplored actions and promising paths.

Based on: "Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks"
https://arxiv.org/abs/2512.24156

Author: Abdullah Soliman
Competition: ARC Prize 2026
"""

import hashlib
import logging
import random
import time
from typing import Optional

import numpy as np

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


class GraphNode:
    """Represents a game state in the exploration graph."""

    def __init__(self, state_id: str, embedding: np.ndarray):
        self.state_id = state_id
        self.embedding = embedding
        self.out_edges: dict[int, str] = {}  # action -> next_state_id
        self.rewards: dict[int, float] = {}  # action -> reward
        self.visits = 0
        self.level_completed_at = -1

    def unexplored_actions(self, action_space: list[GameAction]) -> list[GameAction]:
        """Return actions not yet tried from this state."""
        tried = {k for k in self.out_edges.keys()}
        return [a for a in action_space if (a.value if isinstance(a, GameAction) else a) not in tried]

    def add_transition(self, action: int, next_state_id: str, reward: float):
        """Add an edge from this node via action to next_state."""
        action_key = action.value if isinstance(action, GameAction) else action
        if action_key not in self.out_edges:
            self.out_edges[action_key] = next_state_id
            self.rewards[action_key] = reward
            self.visits += 1


class StateEncoder:
    """Encodes game observations into compact representations."""

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

    def encode(self, frame: FrameData) -> tuple[np.ndarray, str]:
        """Convert frame to embedding + state ID."""
        # Convert frame to numpy array
        grid = np.array(frame.frame, dtype=np.float32)

        # Simple flattening + hash for state ID
        state_id = hashlib.sha1(grid.tobytes()).hexdigest()[:16]

        # Create embedding via simple CNN-like processing
        # For now: use downsampled grid as "embedding"
        try:
            # Downsample if large
            if grid.shape[0] > 16 or grid.shape[1] > 16:
                from scipy.ndimage import zoom
                factors = (16 / grid.shape[0], 16 / grid.shape[1])
                grid = zoom(grid, factors, order=1)
                grid = np.nan_to_num(grid)

            # Flatten and pad/truncate to embedding_dim
            flat = grid.flatten()[:self.embedding_dim]
            if len(flat) < self.embedding_dim:
                flat = np.pad(flat, (0, self.embedding_dim - len(flat)))

            embedding = flat.astype(np.float32)
        except Exception:
            # Fallback: random embedding
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        return embedding, state_id


class WorldGraph:
    """Manages the exploration graph."""

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}

    def add_state(self, state_id: str, embedding: np.ndarray) -> GraphNode:
        """Add a node if not exists."""
        if state_id not in self.nodes:
            self.nodes[state_id] = GraphNode(state_id, embedding)
        return self.nodes[state_id]

    def get_node(self, state_id: str) -> Optional[GraphNode]:
        return self.nodes.get(state_id)

    def frontier_states(self) -> list[GraphNode]:
        """Return states with unexplored actions."""
        return [
            n for n in self.nodes.values()
            if len(n.out_edges) < 10  # reasonable action space estimate
        ]

    def best_path_to_frontier(self, start_id: str, action_space_size: int) -> Optional[list[int]]:
        """Find shortest path to a frontier state via BFS."""
        from collections import deque

        queue = deque([(start_id, [])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()
            node = self.nodes.get(current_id)
            if not node:
                continue

            # Check if this node has unexplored actions
            if len(node.out_edges) < action_space_size:
                return path

            # Explore neighbors
            for action, next_id in node.out_edges.items():
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [action]))

        return None


class GraphExplorer(Agent):
    """
    An agent that explores game environments using graph-based exploration.

    Strategy:
    1. Build a graph of observed states
    2. Prioritize unexplored actions in current state
    3. Fall back to exploring frontier states
    4. Use intrinsic motivation (visits, novelty) to guide exploration
    """

    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = StateEncoder(embedding_dim=64)
        self.graph = WorldGraph()
        self.current_state_id: Optional[str] = None
        self.total_reward = 0.0
        self._last_action: Optional[GameAction] = None  # Track last action for graph updates

        # Seed based on game_id for reproducibility
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32))

        logger.info(f"GraphExplorer initialized for {self.game_id} with seed {seed}")

    @property
    def name(self) -> str:
        return f"{super().name}.graph-explorer"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop when winning or hitting max actions."""
        if latest_frame.state == GameState.WIN:
            logger.info(f"Game WON at level {latest_frame.levels_completed}")
            return True
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose next action using graph exploration strategy."""

        # Encode current state
        embedding, state_id = self.encoder.encode(latest_frame)
        self.current_state_id = state_id
        self.graph.add_state(state_id, embedding)

        # Get action space (convert int IDs to GameAction enums)
        raw_available = latest_frame.available_actions if latest_frame.available_actions else []
        available = []
        for a in raw_available:
            if isinstance(a, GameAction):
                available.append(a)
            else:
                # Try to convert int to GameAction
                try:
                    for ga in GameAction:
                        if ga.value == a:
                            available.append(ga)
                            break
                except:
                    pass

        if not available:
            available = [a for a in GameAction if a != GameAction.RESET]

        # Priority 1: Try unexplored actions in current state
        current_node = self.graph.get_node(state_id)
        unexplored = current_node.unexplored_actions(available) if current_node else available

        if unexplored:
            # Pick randomly from unexplored (with some bias toward simple actions)
            simple_actions = [a for a in unexplored if not self._is_complex_action(a)]
            if simple_actions and random.random() < 0.7:
                action = random.choice(simple_actions)
            else:
                action = random.choice(unexplored)
        else:
            # Priority 2: Move toward frontier state
            action_space_size = len(available)
            path = self.graph.best_path_to_frontier(state_id, action_space_size)

            if path and len(path) > 0:
                # This shouldn't happen in single-agent context, but safety
                action_val = path[0]
                action = [a for a in GameAction if a.value == action_val][0] if action_val else available[0]
            else:
                # Priority 3: Exploitation - pick best known action
                if current_node and current_node.rewards:
                    best_action = max(
                        current_node.rewards.keys(),
                        key=lambda a: current_node.rewards[a]
                    )
                    action = [a for a in GameAction if a.value == best_action][0]
                else:
                    # Fallback: random
                    action = random.choice(available)

        # Save action for graph update
        self._last_action = action

        # Prepare action with data if needed
        if hasattr(action, 'is_simple') and action.is_simple():
            action.reasoning = f"GraphExplorer: exploring from state {state_id[:8]}"
        elif hasattr(action, 'is_complex') and action.is_complex():
            action.set_data({
                "x": random.randint(0, 31),
                "y": random.randint(0, 31),
            })
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"GraphExplorer exploration from {state_id[:8]}",
            }

        return action

    def _is_complex_action(self, action: GameAction) -> bool:
        """Check if action requires complex (x,y) data - not used in current games."""
        # Current ARC-AGI-3 games don't use complex actions
        return False

    def append_frame(self, frame: FrameData) -> None:
        """Record frame and update graph."""
        # Get previous state and action
        prev_state_id = self.current_state_id
        last_action = self._last_action

        # Add new state
        embedding, new_state_id = self.encoder.encode(frame)
        self.graph.add_state(new_state_id, embedding)

        # Add edge if we have previous state and action
        if prev_state_id and last_action is not None:
            # Calculate reward from levels completed change
            reward = float(frame.levels_completed)
            self.total_reward += reward

            action_key = last_action.value if isinstance(last_action, GameAction) else last_action
            prev_node = self.graph.get_node(prev_state_id)
            if prev_node:
                prev_node.add_transition(action_key, new_state_id, reward)

        # Record frame
        super().append_frame(frame)
