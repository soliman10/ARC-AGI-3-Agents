"""
ClickHeuristic Agent for ARC-AGI-3

An agent that:
1. Parses the grid to understand game state
2. Uses simple heuristics based on tile colors
3. Clicks strategically on interesting positions
4. Tracks which clicks lead to progress

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


class GridAnalyzer:
    """Analyze grid state and find interesting positions to click."""

    @staticmethod
    def parse_grid(frame: FrameData) -> np.ndarray:
        """Convert frame to numpy array."""
        if not frame.frame:
            return np.zeros((64, 64), dtype=np.int32)
        return np.array(frame.frame[0], dtype=np.int32)

    @staticmethod
    def find_click_targets(grid: np.ndarray, max_clicks: int = 20) -> list[tuple[int, int]]:
        """Find interesting positions to click based on grid analysis."""
        h, w = grid.shape
        targets = []

        # Find edges between different colors (likely click targets)
        for y in range(h):
            for x in range(w):
                current = grid[y, x]
                if current == 0:  # Skip background
                    continue

                # Check neighbors for color differences
                neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbors.append(grid[ny, nx])

                # If there are neighbors with different colors, this is interesting
                if any(n != current and n != 0 for n in neighbors):
                    targets.append((x, y))

                # Also add some random non-background cells
                if len(targets) < max_clicks and current != 0:
                    targets.append((x, y))

        # Deduplicate and limit
        targets = list(set(targets))[:max_clicks]

        # Add center-weighted sampling for unexplored areas
        if len(targets) < max_clicks:
            center_x, center_y = w // 2, h // 2
            for _ in range(max_clicks - len(targets)):
                # Bias toward center initially, then expand outward
                r = random.random() * min(center_x, center_y)
                angle = random.uniform(0, 2 * 3.14159)
                x = int(center_x + r * np.cos(angle))
                y = int(center_y + r * np.sin(angle))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                targets.append((x, y))

        return targets


class ClickHeuristicAgent(Agent):
    """
    Agent that:
    1. Analyzes grid state
    2. Clicks on strategic positions
    3. Learns which positions lead to wins
    """

    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_click_index = 0
        self._click_targets: list[tuple[int, int]] = []
        self._best_solution: list[tuple[int, int]] = []
        self._click_history: list[tuple[int, int]] = []
        self._level_start_clicks = 0
        self._initial_grid_hash: Optional[str] = None

        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        logger.info(f"ClickHeuristicAgent initialized for {self.game_id}")

    @property
    def name(self) -> str:
        return f"{super().name}.click-heuristic"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop when winning."""
        return latest_frame.state == GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Choose click target."""

        # Check if we need new click targets
        current_grid = GridAnalyzer.parse_grid(latest_frame)
        current_hash = hashlib.md5(current_grid.tobytes()).hexdigest()

        # Reset targets if this is a new level
        if self._initial_grid_hash is None or current_hash != self._initial_grid_hash:
            self._initial_grid_hash = current_hash
            self._click_targets = GridAnalyzer.find_click_targets(current_grid)
            self._last_click_index = 0
            self._level_start_clicks = len(self._click_history)
            logger.info(f"New level detected, {len(self._click_targets)} targets")

        # Get next target
        if self._last_click_index < len(self._click_targets):
            x, y = self._click_targets[self._last_click_index]
            self._last_click_index += 1
        else:
            # All targets tried, try random
            h, w = current_grid.shape
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

        self._click_history.append((x, y))

        # Create ACTION6 with coordinates
        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        action.reasoning = {"target": f"({x},{y})", "index": self._last_click_index}

        return action

    def append_frame(self, frame: FrameData) -> None:
        """Track progress."""
        # If we won, save the solution
        if frame.state == GameState.WIN:
            clicks_this_level = self._click_history[self._level_start_clicks:]
            logger.info(f"Level won with {len(clicks_this_level)} clicks: {clicks_this_level}")
            self._best_solution = clicks_this_level

        super().append_frame(frame)


class SmartExplorerAgent(Agent):
    """
    Combines:
    1. Grid analysis for finding click targets
    2. Graph-based state tracking
    3. Learning from successful click patterns
    """

    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._click_targets: list[tuple[int, int]] = []
        self._click_index = 0
        self._visited_states: dict[str, int] = {}
        self._best_score = 0
        self._current_level_start = 0
        self._initial_hash: Optional[str] = None

        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        logger.info(f"SmartExplorerAgent initialized for {self.game_id}")

    @property
    def name(self) -> str:
        return f"{super().name}.smart-explorer"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state == GameState.WIN

    def _get_grid_hash(self, frame: FrameData) -> str:
        """Get a hash of the current grid."""
        grid = np.array(frame.frame[0], dtype=np.int32) if frame.frame else np.zeros((64, 64))
        return hashlib.md5(grid.tobytes()).hexdigest()

    def _find_targets(self, frame: FrameData) -> list[tuple[int, int]]:
        """Find click targets using grid analysis."""
        if not frame.frame:
            return []
        grid = np.array(frame.frame[0], dtype=np.int32)
        h, w = grid.shape

        targets = []

        # Find color boundaries - these are often interactive
        for y in range(h):
            for x in range(w):
                val = grid[y, x]
                if val == 0:
                    continue

                # Check if this is a boundary pixel
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if grid[ny, nx] != val and grid[ny, nx] != 0:
                            targets.append((x, y))
                            break

        # If not enough boundary pixels, sample throughout
        if len(targets) < 10:
            for _ in range(20):
                x = random.randint(0, w - 1)
                y = random.randint(0, h - 1)
                if grid[y, x] != 0:
                    targets.append((x, y))

        return list(set(targets))[:30]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        """Smart action selection."""

        grid_hash = self._get_grid_hash(latest_frame)

        # Check if new level
        if self._initial_hash != grid_hash:
            self._initial_hash = grid_hash
            self._click_targets = self._find_targets(latest_frame)
            self._click_index = 0
            self._current_level_start = len(frames)
            logger.info(f"New level: {len(self._click_targets)} targets")

        # Track state visits for novelty bonus
        visits = self._visited_states.get(grid_hash, 0)
        self._visited_states[grid_hash] = visits + 1

        # Select click target
        if self._click_index < len(self._click_targets):
            x, y = self._click_targets[self._click_index]
            self._click_index += 1
        else:
            # Random exploration
            if latest_frame.frame:
                h, w = np.array(latest_frame.frame[0]).shape
            else:
                h, w = 64, 64
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

        # Create action
        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        action.reasoning = {"target": f"({x},{y})", "visits": visits}

        return action

    def append_frame(self, frame: FrameData) -> None:
        super().append_frame(frame)
        if frame.state == GameState.WIN:
            logger.info(f"WON! Score: {frame.levels_completed}")
