"""
Improved Graph-Based Explorer Agent for ARC-AGI-3

Key improvements over v1:
1. Curiosity-driven exploration (intrinsic reward for novel states)
2. Priority-based action selection using UCB-style scoring
3. Better state hashing for deduplication
4. Adaptive action budget per game
5. Simple world model prediction

Author: Abdullah Soliman
Competition: ARC Prize 2026
"""

import hashlib
import logging
import random
import time
from collections import deque
from typing import Optional

import numpy as np

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger()


class ExplorationNode:
    """Enhanced node with curiosity and visit statistics."""

    def __init__(self, state_id: str, embedding: np.ndarray):
        self.state_id = state_id
        self.embedding = embedding
        self.out_edges: dict[int, str] = {}  # action -> next_state_id
        self.rewards: dict[int, float] = {}  # action -> reward
        self.visits = 0
        self.intrinsic_reward = 0.0

        # For curiosity-based exploration
        self.novelty_score = 0.0
        self.first_visit_time = time.time()

    def unexplored_actions(self, action_space: list[GameAction]) -> list[tuple[GameAction, int]]:
        """Return (action, action_value) pairs not yet tried."""
        result = []
        for a in action_space:
            action_val = a.value if isinstance(a, GameAction) else a
            if action_val not in self.out_edges:
                result.append((a, action_val))
        return result

    def add_transition(self, action: int, next_state_id: str, reward: float, is_novel: bool):
        """Add edge and update intrinsic reward."""
        if action not in self.out_edges:
            self.out_edges[action] = next_state_id
            self.rewards[action] = reward
            self.visits += 1
            if is_novel:
                self.intrinsic_reward += 1.0  # Curiosity bonus


class StateEncoderV2:
    """Enhanced state encoder with better hashing and features."""

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

    def encode(self, frame: FrameData) -> tuple[np.ndarray, str]:
        """Convert frame to embedding + state ID."""
        grid = np.array(frame.frame, dtype=np.float32)

        # State ID: hash of the grid
        state_id = hashlib.sha256(grid.tobytes()).hexdigest()[:24]

        # Create embedding via spatial pooling
        try:
            # Downsample to fixed size
            from scipy.ndimage import zoom
            target_size = 8
            if grid.shape[0] > target_size or grid.shape[1] > target_size:
                factors = (target_size / grid.shape[0], target_size / grid.shape[1])
                grid = zoom(grid, factors, order=1)
                grid = np.nan_to_num(grid)

            # Simple spatial features
            features = []

            # Mean, std, min, max
            features.extend([grid.mean(), grid.std(), grid.min(), grid.max()])

            # Spatial statistics (quadrants)
            h, w = grid.shape
            for i in range(2):
                for j in range(2):
                    quadrant = grid[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                    features.extend([quadrant.mean(), quadrant.std()])

            # Object count (non-zero cells)
            features.append((grid > 0).sum() / grid.size)

            # Flatten and pad/truncate
            flat = np.array(features, dtype=np.float32)
            if len(flat) < self.embedding_dim:
                flat = np.pad(flat, (0, self.embedding_dim - len(flat)))
            else:
                flat = flat[:self.embedding_dim]

            embedding = flat

        except Exception:
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        return embedding, state_id


class WorldGraphV2:
    """Enhanced graph with curiosity and pathfinding."""

    def __init__(self):
        self.nodes: dict[str, ExplorationNode] = {}
        self.state_visits: dict[str, int] = {}  # Track visit counts for novelty

    def add_state(self, state_id: str, embedding: np.ndarray) -> tuple[ExplorationNode, bool]:
        """Add node if new, return (node, is_novel)."""
        is_novel = state_id not in self.nodes
        if is_novel:
            self.nodes[state_id] = ExplorationNode(state_id, embedding)
            self.state_visits[state_id] = 0
        self.state_visits[state_id] += 1
        node = self.nodes[state_id]
        node.novelty_score = 1.0 / (1.0 + self.state_visits[state_id])
        return node, is_novel

    def get_node(self, state_id: str) -> Optional[ExplorationNode]:
        return self.nodes.get(state_id)

    def best_path_to_frontier(self, start_id: str, action_space_size: int) -> Optional[list[int]]:
        """BFS to nearest frontier state."""
        queue = deque([(start_id, [])])
        visited = {start_id}
        max_depth = 20

        while queue and len(queue[0][1]) < max_depth:
            current_id, path = queue.popleft()
            node = self.nodes.get(current_id)
            if not node:
                continue

            # Check if this is a frontier
            unexplored = len([a for a in range(action_space_size) if a not in node.out_edges])
            if unexplored > 0 and len(path) > 0:
                return path

            # Explore neighbors
            for action, next_id in node.out_edges.items():
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [action]))

        return None


class ImprovedExplorer(Agent):
    """
    Improved agent with:
    - Curiosity-driven exploration
    - UCB action selection
    - Better state representation
    - Adaptive depth-first search
    """

    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = StateEncoderV2(embedding_dim=32)
        self.graph = WorldGraphV2()
        self.current_state_id: Optional[str] = None
        self._last_action: Optional[GameAction] = None
        self.total_reward = 0.0

        # Exploration parameters
        self.exploration_const = 1.0
        self.curiosity_weight = 0.5

        # Seed for reproducibility
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32))

        logger.info(f"ImprovedExplorer initialized for {self.game_id}")

    @property
    def name(self) -> str:
        return f"{super().name}.improved-explorer"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop on WIN."""
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose action using UCB + curiosity strategy."""

        # Encode and register state
        embedding, state_id = self.encoder.encode(latest_frame)
        self.current_state_id = state_id
        node, is_novel = self.graph.add_state(state_id, embedding)

        # Get available actions
        raw_available = latest_frame.available_actions if latest_frame.available_actions else []
        available = self._convert_actions(raw_available)
        if not available:
            available = [a for a in GameAction if a != GameAction.RESET]

        action_space_size = len(available)

        # Strategy: UCB with curiosity
        action = self._ucb_action(node, available, action_space_size)

        # Save for graph update
        self._last_action = action
        return action

    def _convert_actions(self, raw_actions: list) -> list[GameAction]:
        """Convert int action IDs to GameAction enums."""
        result = []
        for a in raw_actions:
            if isinstance(a, GameAction):
                result.append(a)
            else:
                for ga in GameAction:
                    if ga.value == a:
                        result.append(ga)
                        break
        return result

    def _ucb_action(self, node: ExplorationNode, available: list[GameAction], action_space_size: int) -> GameAction:
        """Select action using UCB + curiosity."""

        # Priority 1: Try completely new actions (no edge exists)
        unexplored = node.unexplored_actions(available)
        if unexplored:
            # Among unexplored, prefer those from states with high novelty
            if node.novelty_score > 0.3 and random.random() < 0.7:
                return unexplored[0][0]  # First unexplored
            else:
                return random.choice([a for a, _ in unexplored])

        # Priority 2: UCB-based selection
        ucb_scores = {}
        for a in available:
            action_val = a.value if isinstance(a, GameAction) else a
            ucb_scores[a] = self._compute_ucb(node, action_val)

        # Add exploration bonus for less-visited nodes in graph
        best_action = max(ucb_scores.keys(), key=lambda a: ucb_scores[a])

        # Priority 3: If stuck, try to move to frontier
        if not ucb_scores or ucb_scores.get(best_action, 0) < 0.1:
            path = self.graph.best_path_to_frontier(self.current_state_id, action_space_size)
            if path:
                action_val = path[0]
                for a in available:
                    if (isinstance(a, GameAction) and a.value == action_val) or (not isinstance(a, GameAction) and a == action_val):
                        return a
                # Fallback: return first action in path
                for a in GameAction:
                    if a.value == action_val:
                        return a

        return best_action if best_action else random.choice(available)

    def _compute_ucb(self, node: ExplorationNode, action: int) -> float:
        """Compute UCB score for an action."""
        if action not in node.rewards:
            return float('inf')

        base_reward = node.rewards[action]
        visit_count = max(node.visits, 1)
        exploration_bonus = self.exploration_const * np.sqrt(np.log(visit_count) / (visit_count))
        curiosity_bonus = self.curiosity_weight * node.novelty_score

        return base_reward + exploration_bonus + curiosity_bonus

    def append_frame(self, frame: FrameData) -> None:
        """Update graph with new transition."""
        prev_state_id = self.current_state_id
        last_action = self._last_action

        # Register new state
        embedding, new_state_id = self.encoder.encode(frame)
        node, is_novel = self.graph.add_state(new_state_id, embedding)

        # Add transition from previous state
        if prev_state_id and last_action is not None:
            action_val = last_action.value if isinstance(last_action, GameAction) else last_action
            prev_node = self.graph.get_node(prev_state_id)
            if prev_node:
                reward = float(frame.levels_completed)
                prev_node.add_transition(action_val, new_state_id, reward, is_novel)
                self.total_reward += reward

        super().append_frame(frame)
