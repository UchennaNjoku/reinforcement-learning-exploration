"""
Curriculum Learning for GridWorld Navigation
=============================================
Defines training phases that progressively increase difficulty
by varying wall length. The agent's weights carry forward between
phases, with a partial epsilon reset to encourage re-exploration.

Phases advance when evaluation success rate exceeds a threshold
for consecutive checks — not on a fixed episode count.

Key insight: reward shaping helps early (learn goal-seeking) but
hurts later (penalizes necessary detours around walls). Each phase
has a shaping_weight that controls how much distance-based shaping
is applied. Later phases reduce or eliminate it.

Author: Chenna (CS Senior, Bethune-Cookman University)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Phase:
    """A single curriculum phase."""
    name: str
    wall_length: int
    max_episodes: int           # hard cap — move on even if threshold not met
    advance_threshold: float    # success rate to advance (e.g. 0.85 = 85%)
    consecutive_required: int   # how many consecutive evals must meet threshold
    eps_reset: float            # epsilon to reset to at start of this phase
    shaping_weight: float       # how much distance shaping to apply (1.0=full, 0.0=none)


@dataclass
class CurriculumConfig:
    """
    Full curriculum definition.

    Attributes:
        phases: ordered list of training phases (easy → hard)
        eval_episodes: episodes per evaluation check
        eval_interval: training episodes between evaluation checks
    """
    phases: List[Phase]
    eval_episodes: int = 25
    eval_interval: int = 250

    def __post_init__(self):
        if not self.phases:
            raise ValueError("Curriculum must have at least one phase")

    @property
    def total_max_episodes(self) -> int:
        return sum(p.max_episodes for p in self.phases)

    def summary(self) -> str:
        lines = []
        for i, p in enumerate(self.phases, 1):
            lines.append(
                f"  Phase {i}: {p.name:<20s} │ "
                f"wall={p.wall_length} │ "
                f"max_eps={p.max_episodes:>5d} │ "
                f"advance@{p.advance_threshold*100:.0f}% "
                f"(×{p.consecutive_required}) │ "
                f"ε={p.eps_reset:.2f} │ "
                f"shaping={p.shaping_weight:.1f}"
            )
        return "\n".join(lines)


@dataclass
class PhaseTracker:
    """
    Tracks progress within a curriculum phase.
    Decides when to advance based on consecutive eval successes.
    """
    phase: Phase
    episodes_completed: int = 0
    consecutive_passes: int = 0
    best_success_rate: float = 0.0
    total_evals: int = 0

    def record_eval(self, success_rate: float) -> bool:
        """
        Record an evaluation result.
        Returns True if the phase should advance (threshold met enough times).
        """
        self.total_evals += 1
        self.best_success_rate = max(self.best_success_rate, success_rate)

        if success_rate >= self.phase.advance_threshold:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        return self.consecutive_passes >= self.phase.consecutive_required

    def should_force_advance(self) -> bool:
        """Check if we've hit the hard episode cap."""
        return self.episodes_completed >= self.phase.max_episodes

    @property
    def status_str(self) -> str:
        return (
            f"Phase '{self.phase.name}' │ "
            f"ep={self.episodes_completed}/{self.phase.max_episodes} │ "
            f"best_sr={self.best_success_rate*100:.1f}% │ "
            f"passes={self.consecutive_passes}/{self.phase.consecutive_required}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Default Curriculum
# ─────────────────────────────────────────────────────────────────────────────
def default_curriculum() -> CurriculumConfig:
    """
    Standard 4-phase curriculum for 12×12 GridWorld.

    Phase 1 — No wall:      Full shaping, learn goal-seeking
    Phase 2 — Short wall:   Reduced shaping, learn walls exist
    Phase 3 — Medium wall:  Minimal shaping, learn detours
    Phase 4 — Full wall:    Low shaping, mostly sparse reward
    """
    return CurriculumConfig(
        phases=[
            Phase(
                name="No Wall",
                wall_length=0,
                max_episodes=3000,
                advance_threshold=0.80,
                consecutive_required=2,
                eps_reset=1.0,
                shaping_weight=1.0,
            ),
            Phase(
                name="Short Wall",
                wall_length=2,
                max_episodes=5000,
                advance_threshold=0.70,
                consecutive_required=2,
                eps_reset=0.9,
                shaping_weight=1.0,
            ),
            Phase(
                name="Medium Wall",
                wall_length=3,
                max_episodes=6000,
                advance_threshold=0.60,
                consecutive_required=2,
                eps_reset=0.6,
                shaping_weight=0.95,
            ),
            Phase(
                name="Full Wall",
                wall_length=4,
                max_episodes=10000,
                advance_threshold=0.65,
                consecutive_required=3,
                eps_reset=0.3,
                shaping_weight=0.8,
            ),
        ],
        eval_episodes=25,
        eval_interval=250,
    )


def aggressive_curriculum() -> CurriculumConfig:
    """
    Faster curriculum with fewer episodes per phase.
    Good for quick iteration / testing.
    """
    return CurriculumConfig(
        phases=[
            Phase(
                name="No Wall",
                wall_length=0,
                max_episodes=2000,
                advance_threshold=0.75,
                consecutive_required=2,
                eps_reset=1.0,
                shaping_weight=1.0,
            ),
            Phase(
                name="Short Wall",
                wall_length=2,
                max_episodes=3000,
                advance_threshold=0.65,
                consecutive_required=2,
                eps_reset=0.5,
                shaping_weight=0.3,
            ),
            Phase(
                name="Full Wall",
                wall_length=4,
                max_episodes=6000,
                advance_threshold=0.60,
                consecutive_required=2,
                eps_reset=0.3,
                shaping_weight=0.1,
            ),
        ],
        eval_episodes=25,
        eval_interval=200,
    )