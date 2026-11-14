"""The GELLO teleop client: its arithmetic, and its behaviour against a scripted Env."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from xpisode_env.env import Observation, ObservedSample
from xpisode_env.messages import GroupState, Tick
from xpisode_format.documents import Prompts

from xpisode_abc_gello.teleop import (
    Converge,
    Feedback,
    Follow,
    Gripper,
    Mode,
    Teleop,
    TeleopConfig,
    grip_torque,
)

# --- converging joints ----------------------------------------------------------------------


def test_takeover_does_not_jump() -> None:
    join = Converge(rate=0.5, still=0.003)
    leader, follower = np.array([0.0, 1.0]), np.array([0.3, 0.8])
    join.anchor(leader, follower)
    np.testing.assert_allclose(join.step(leader), follower)


def test_a_still_or_jittering_leader_leaves_the_follower_still() -> None:
    join = Converge(rate=0.5, still=0.003)
    leader, follower = np.zeros(6), np.full(6, 0.4)
    join.anchor(leader, follower)
    rng = np.random.default_rng(0)
    for _ in range(1000):  # ten seconds of two-count jitter
        target = join.step(leader + rng.uniform(-0.0015, 0.0015, 6))
    np.testing.assert_allclose(join.gap, 0.4)
    assert np.all(np.abs(target - follower) < 0.002)


def test_the_gap_closes_with_motion_and_never_overshoots() -> None:
    join = Converge(rate=0.5, still=0.003)
    join.anchor(np.zeros(1), np.array([0.2]))
    positions = np.r_[np.linspace(0, 0.5, 51), np.linspace(0.5, 0, 51)]  # out and back
    for q in positions:
        join.step(np.array([q]))
    assert join.gap[0] == 0.0
    target = join.step(np.array([0.1]))
    assert target[0] == pytest.approx(0.1)  # one to one again


def test_follower_moves_at_most_one_and_a_half_times_the_leader() -> None:
    join = Converge(rate=0.5, still=0.0)
    join.anchor(np.zeros(1), np.array([-1.0]))
    previous = join.step(np.zeros(1))
    for q in np.linspace(0.01, 0.5, 50):
        target = join.step(np.array([q]))
        assert abs(target[0] - previous[0]) <= 1.5 * 0.01 + 1e-9
        previous = target


def test_rate_zero_is_abcs_fixed_offset() -> None:
    join = Converge(rate=0.0, still=0.003)
    join.anchor(np.zeros(1), np.array([0.3]))
    for q in np.linspace(0, 1, 100):
        target = join.step(np.array([q]))
    assert target[0] == pytest.approx(1.3)


# --- abc's gripper rule ---------------------------------------------------------------------


def test_gripper_relative_then_snaps_open() -> None:
    grip = Gripper()
    grip.anchor(opening=0.5, follower=0.2)  # policy had it nearly closed
    assert grip.step(0.5) == pytest.approx(0.2)
    assert grip.step(0.55) == pytest.approx(0.25)
    assert grip.step(0.61) == 1.0  # opened 0.1 past the start: open fully
    assert grip.step(0.65) == 1.0
    assert grip.step(0.58) == pytest.approx(0.58)  # below the floor (0.6): one to one


def test_gripper_squeezed_fully_means_closed() -> None:
    grip = Gripper()
    grip.anchor(opening=0.3, follower=0.6)
    assert grip.step(0.1) == pytest.approx(0.4)
    assert grip.step(0.0) == 0.0
    assert grip.step(0.2) == pytest.approx(0.2)


# --- the grip felt --------------------------------------------------------------------------

FEEDBACK = Feedback(
    target="leader/trigger_feedback/trigger_torque",
    effort="follower/state/gripper_eff",
    velocity="follower/state/gripper_vel",
)


def test_grip_torque_from_the_recorded_numbers() -> None:
    assert grip_torque(0.2, 0.0, FEEDBACK) == 0.0  # open and still: noise
    assert grip_torque(1.03, 0.0, FEEDBACK) == pytest.approx(0.073)  # holding something
    assert grip_torque(2.4, 1.0, FEEDBACK) == 0.0  # closing fast: acceleration
    assert grip_torque(-1.0, 0.0, FEEDBACK) == 0.0  # pushing open is no grip


# --- the client against a scripted Env ------------------------------------------------------

JOINTS = Follow(
    target="follower/joint_position/joint_pos",
    source="leader/input/joint_pos",
    state="follower/state/joint_pos",
)
GRIPPER = Follow(
    target="follower/joint_position/gripper_pos",
    source="leader/input/trigger",
    state="follower/state/gripper_pos",
    scale=-1.0,
    offset=1.0,
    takeover="gripper",
)
CONFIG = TeleopConfig(
    group="arm", follow=(JOINTS, GRIPPER), leader_group="leader", feedback=FEEDBACK
)


class ScriptedEnv:
    def __init__(self) -> None:
        self.acts: list[tuple[str, dict[str, Any], int]] = []

    def act(self, group: str, values: dict[str, Any], *, start_tick: int, observation: Any) -> int:
        self.acts.append((group, {k: np.asarray(v) for k, v in values.items()}, start_tick))
        return len(self.acts)


def observe(
    tick: int,
    *,
    holder: str | None,
    generation: int,
    leader: list[float],
    trigger: float = 0.0,
    follower: list[float] | None = None,
    opening: float = 1.0,
    effort: float = 0.0,
    gripper_vel: float = 0.0,
    leader_age_s: float = 0.005,
) -> Observation:
    def sample(age_s: float, **values: Any) -> ObservedSample:
        arrays = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in values.items()}
        return ObservedSample(0, tick, int(age_s * 1e9), arrays)

    return Observation(
        client="teleop",
        tick=Tick(
            tick=tick,
            time_ns=tick * 10_000_000,
            groups={
                "arm": GroupState(holder=holder, generation=generation),
                "leader": GroupState(holder="teleop", generation=1),
            },
        ),
        samples={
            "leader/input": sample(leader_age_s, joint_pos=leader, trigger=[trigger]),
            "follower/state": sample(
                0.005,
                joint_pos=follower if follower is not None else leader,
                gripper_pos=[opening],
                gripper_eff=[effort],
                gripper_vel=[gripper_vel],
            ),
        },
        prompts=Prompts(),
    )


def test_takeover_follows_without_a_jump_and_feeds_the_grip_back() -> None:
    env = ScriptedEnv()
    client = Teleop(env, CONFIG)  # type: ignore[arg-type]
    # A policy drives: nothing for the follower, zero on the trigger.
    client.handle(observe(1, holder="policy", generation=1, leader=[0.0] * 6, follower=[0.2] * 6))
    assert client.mode is Mode.IDLE
    assert [a[0] for a in env.acts] == ["leader"]
    assert env.acts[0][1]["leader/trigger_feedback/trigger_torque"][0, 0] == 0.0
    # Granted: the follower stays where the policy left it.
    env.acts.clear()
    client.handle(
        observe(2, holder="teleop", generation=2, leader=[0.0] * 6, follower=[0.2] * 6, effort=1.03)
    )
    assert client.mode is Mode.FOLLOWING
    arm = next(a for a in env.acts if a[0] == "arm")
    np.testing.assert_allclose(arm[1]["follower/joint_position/joint_pos"][0], [0.2] * 6)
    assert arm[2] == 3
    leader = next(a for a in env.acts if a[0] == "leader")
    assert leader[1]["leader/trigger_feedback/trigger_torque"][0, 0] == pytest.approx(0.073)


def test_a_stale_leader_is_not_followed() -> None:
    env = ScriptedEnv()
    client = Teleop(env, CONFIG)  # type: ignore[arg-type]
    client.handle(observe(1, holder="teleop", generation=1, leader=[0.0] * 6, leader_age_s=0.5))
    assert [a[0] for a in env.acts] == ["leader"]


def test_losing_control_stops_following_and_the_grip() -> None:
    env = ScriptedEnv()
    client = Teleop(env, CONFIG)  # type: ignore[arg-type]
    client.handle(observe(1, holder="teleop", generation=1, leader=[0.0] * 6, effort=1.0))
    env.acts.clear()
    client.handle(observe(2, holder="policy", generation=2, leader=[0.0] * 6, effort=1.0))
    assert client.mode is Mode.IDLE
    assert [a[0] for a in env.acts] == ["leader"]
    assert env.acts[0][1]["leader/trigger_feedback/trigger_torque"][0, 0] == 0.0


def test_configuration_mistakes() -> None:
    with pytest.raises(ValidationError, match="feedback needs leader_group"):
        TeleopConfig(group="arm", follow=(JOINTS,), feedback=FEEDBACK)
    with pytest.raises(ValidationError, match="at most one"):
        TeleopConfig(group="arm", follow=(GRIPPER, GRIPPER))
    assert CONFIG.groups() == ("arm", "leader")
    assert CONFIG.streams() == ("leader/input", "follower/state")
