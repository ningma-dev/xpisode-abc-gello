"""Teleoperation from a GELLO, a leader that cannot be driven, with the follower's grip felt.

xpisode's reference client (``xpisode_env.teleop``) keeps a leader and follower together by
driving the leader; a GELLO cannot be driven, so when this client is given the follower (a
takeover, as in DAgger) the two are apart. abc (``deploy/robot/gym/dagger_rollout.py``) moves
the follower's end effector by the leader's motion from where each was at the takeover, through
IK, and stays apart. This client does it in joint space and closes the gap:

- **Taking over** it notes, per joint, ``gap = follower - leader``; the follower's target is the
  leader plus the gap, so nothing jumps.
- **The gap closes only as the leader moves**: each tick it shrinks by ``converge_rate`` times
  how far the leader moved (beyond ``still_rad``, the encoder's jitter), never past zero. A leader
  held still leaves the follower still. Once closed, the mapping is one to one again, and what
  is recorded is the same as from ordinary teleoperation.
- **The gripper** follows abc: by its opening relative to the leader's at the takeover, until
  the operator opens it 0.1 past that (it snaps open, then follows one to one) or squeezes fully.
- **The grip**: while it follows, the follower's gripper effort, beyond ``deadband`` and only
  while the gripper is nearly still (moving, its effort is acceleration, not contact), goes back
  to the GELLO's trigger as ``gain`` times that, as resistance to squeezing.

    python -m xpisode_abc_gello.teleop --station yam_abc --name teleop_left --config gello_left.toml
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Literal

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, model_validator
from xpisode_env.env import Env, EnvClosedError, Observation, connect

FloatArray = npt.NDArray[np.float64]


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Follow(_Config):
    """``target = scale * source + offset``, from the leader, as in xpisode_env.teleop."""

    target: str
    """Follower action part, ``device/endpoint/field``."""
    source: str
    """Leader input, ``device/stream/field``."""
    state: str
    """The follower's measurement of the same quantity, ``device/stream/field``."""
    scale: float = 1.0
    offset: float = 0.0
    takeover: Literal["converge", "gripper"] = "converge"
    """How it joins the follower at a takeover: a closing gap (joints) or abc's gripper rule."""


class Feedback(_Config):
    """The follower's grip, felt on the leader's trigger."""

    target: str
    """Leader action part, ``device/endpoint/field``: the GELLO's trigger torque."""
    effort: str
    """Follower gripper effort, ``device/stream/field``; positive squeezes."""
    velocity: str
    """Follower gripper velocity, ``device/stream/field``."""
    deadband: NonNegativeFloat = 0.3
    """N*m of effort that is no contact: still and open, it stays under 0.2."""
    gain: NonNegativeFloat = 0.1
    """Trigger N*m per N*m of effort past the deadband: a 1 N*m grip gives 0.07."""
    still_speed: PositiveFloat = 0.2
    """Above this gripper speed the effort is not passed on."""


class TeleopConfig(_Config):
    group: str
    """The follower's control group."""
    follow: tuple[Follow, ...]
    leader_group: str | None = None
    """The group holding the leader's trigger_feedback endpoint; ``feedback`` needs it."""
    feedback: Feedback | None = None
    converge_rate: NonNegativeFloat = 0.5
    """How much of the leader's motion closes the gap: 0 never closes it (abc's behaviour)."""
    still_rad: NonNegativeFloat = 0.003
    """Leader motion per tick below this (about two encoder counts) is jitter, not motion."""
    stale_s: PositiveFloat = 0.1
    """A leader sample older than this is not followed: the follower holds."""

    @model_validator(mode="after")
    def _check(self) -> TeleopConfig:
        if (self.feedback is None) != (self.leader_group is None):
            raise ValueError("feedback needs leader_group, and leader_group needs feedback")
        if sum(f.takeover == "gripper" for f in self.follow) > 1:
            raise ValueError("at most one follow entry is the gripper")
        return self

    def groups(self) -> tuple[str, ...]:
        return (self.group,) if self.leader_group is None else (self.group, self.leader_group)

    def streams(self) -> tuple[str, ...]:
        paths = [p for f in self.follow for p in (f.source, f.state)]
        if self.feedback is not None:
            paths += [self.feedback.effort, self.feedback.velocity]
        return tuple(dict.fromkeys(p.rsplit("/", 1)[0] for p in paths))


# --- the arithmetic, without a station ------------------------------------------------------


@dataclass
class Converge:
    """A follow entry whose follower starts ``gap`` away and comes back as the leader moves."""

    rate: float
    still: float
    gap: FloatArray = field(default_factory=lambda: np.zeros(0))
    _last: FloatArray | None = None

    def anchor(self, base: FloatArray, follower: FloatArray) -> None:
        self.gap = np.asarray(follower, dtype=np.float64) - base
        self._last = base.copy()

    def step(self, base: FloatArray) -> FloatArray:
        """``base`` is ``scale * source + offset`` now; returns the target."""
        assert self._last is not None
        moved = np.maximum(np.abs(base - self._last) - self.still, 0.0)
        self._last = base.copy()
        self.gap = np.sign(self.gap) * np.maximum(np.abs(self.gap) - self.rate * moved, 0.0)
        return base + self.gap


@dataclass
class Gripper:
    """abc's ``RelativeGelloController._gripper``, on an opening from 0 (closed) to 1."""

    offset: float = 0.0
    start: float = 0.0
    relative: bool = True
    snap: bool = False
    floor: float = 0.0

    def anchor(self, opening: float, follower: float) -> None:
        self.offset = float(follower) - opening
        self.start = opening
        self.relative = True
        self.snap = False

    def step(self, opening: float) -> float:
        value = float(np.clip(opening, 0.0, 1.0))
        if self.snap:
            if value >= self.floor:
                return 1.0
            self.snap = False
        if self.relative and self.start < 0.8 and value - self.start >= 0.1:
            self.relative, self.snap = False, True
            self.floor = min(1.0, self.start + 0.1)
            return 1.0
        if self.relative:
            if value <= 0.0:  # squeezed fully: the operator means closed
                self.relative = False
                return value
            return float(np.clip(value + self.offset, 0.0, 1.0))
        return value


def grip_torque(effort: float, velocity: float, config: Feedback) -> float:
    """What the trigger should push back with for this follower gripper state."""
    if abs(velocity) > config.still_speed:
        return 0.0
    return config.gain * max(effort - config.deadband, 0.0)


# --- the client -----------------------------------------------------------------------------


class Mode(Enum):
    IDLE = "idle"
    """Somebody else drives the follower, or nobody."""
    FOLLOWING = "following"


def _value(observation: Observation, path: str) -> FloatArray:
    stream, _, name = path.rpartition("/")
    return np.asarray(observation.samples[stream].values[name], dtype=np.float64)


def _age_s(observation: Observation, path: str) -> float:
    return observation.samples[path.rpartition("/")[0]].age_ns / 1e9


class Teleop:
    def __init__(self, env: Env, config: TeleopConfig) -> None:
        self.env = env
        self.config = config
        self.mode = Mode.IDLE
        self._generation: int | None = None
        self._joins: dict[str, Converge | Gripper] = {
            f.target: (
                Gripper() if f.takeover == "gripper" else Converge(config.converge_rate, config.still_rad)
            )
            for f in config.follow
        }

    def _base(self, observation: Observation, follow: Follow) -> FloatArray:
        return follow.scale * _value(observation, follow.source) + follow.offset

    def _anchor(self, observation: Observation) -> None:
        for follow in self.config.follow:
            join = self._joins[follow.target]
            base, state = self._base(observation, follow), _value(observation, follow.state)
            if isinstance(join, Gripper):
                join.anchor(float(base.reshape(-1)[0]), float(state.reshape(-1)[0]))
            else:
                join.anchor(base, state)

    def _targets(self, observation: Observation) -> dict[str, FloatArray]:
        targets = {}
        for follow in self.config.follow:
            join, base = self._joins[follow.target], self._base(observation, follow)
            if isinstance(join, Gripper):
                targets[follow.target] = np.full(base.shape, join.step(float(base.reshape(-1)[0])))
            else:
                targets[follow.target] = join.step(base)
        return targets

    def gaps(self) -> dict[str, FloatArray]:
        """What is left to close, per converging entry (for display and tests)."""
        return {k: j.gap.copy() for k, j in self._joins.items() if isinstance(j, Converge)}

    def handle(self, observation: Observation) -> None:
        config, tick = self.config, observation.tick.tick
        holds = observation.holds(config.group)
        generation = observation.generation(config.group) if holds else None
        if generation != self._generation:  # granted anew, or lost
            self._generation = generation
            if holds:
                self._anchor(observation)
                self.mode = Mode.FOLLOWING
            else:
                self.mode = Mode.IDLE
        fresh = all(_age_s(observation, f.source) < config.stale_s for f in config.follow)
        if self.mode is Mode.FOLLOWING and fresh:
            steps = {k: v[np.newaxis] for k, v in self._targets(observation).items()}
            self.env.act(config.group, steps, start_tick=tick + 1, observation=observation)
        feedback, leader = config.feedback, config.leader_group
        if feedback is not None and leader is not None and observation.holds(leader):
            torque = 0.0
            if self.mode is Mode.FOLLOWING:
                torque = grip_torque(
                    float(_value(observation, feedback.effort).reshape(-1)[0]),
                    float(_value(observation, feedback.velocity).reshape(-1)[0]),
                    feedback,
                )
            self.env.act(
                leader,
                {feedback.target: np.array([[torque]])},
                start_tick=tick + 1,
                observation=observation,
            )

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                observation = self.env.observe(timeout_s=0.5)
            except TimeoutError:
                continue
            except EnvClosedError:
                return
            self.handle(observation)


def load(path: Path) -> TeleopConfig:
    with path.open("rb") as file:
        return TeleopConfig.model_validate(tomllib.load(file))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xpisode_abc_gello.teleop", description=__doc__)
    parser.add_argument("--station", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name", required=True, help="this client's name")
    args = parser.parse_args(argv)
    config = load(args.config)
    env = connect(
        args.name,
        args.station,
        groups=config.groups(),
        streams=config.streams(),
        description="GELLO teleoperation: converging takeover, grip felt on the trigger",
    )
    stop = threading.Event()

    def quit_(signum: int, frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, quit_)
    signal.signal(signal.SIGINT, quit_)
    Teleop(env, config).run(stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
