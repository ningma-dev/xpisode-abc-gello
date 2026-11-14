"""The GELLO's numbers: encoder counts to YAM joint angles, the trigger, and the currents of
its force feedback. Pure functions, ported from abc's ``deploy/robot/leaders/gello_leader.py``
(amazon-far/abc d0832d1, Apache-2.0), so they can be tested without the arm.

Joint angles are in the YAM follower's convention (i2rt's), as abc's leader publishes them.
The trigger is xpisode's leader convention, as YamLeader reports it: 0 released, 1 squeezed;
a teleop client maps it to the follower's opening as ``1 - trigger``. Between those, the
mapping is abc's: the follower opening it would have commanded is ``1 - trigger``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DOF = 6
"""Arm joints; the seventh motor is the trigger."""
MOTORS = DOF + 1

CALIBRATION_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.357])
"""Where the GELLO must be when it opens: arm joints at the YAM's zero, trigger released."""
CALIBRATION_STEP = np.pi / 4
"""An offset is a whole number of these: the horns go on in 45-degree steps."""

LEADER_GRIPPER_RANGE = (-0.6040225, 0.33736414)
"""Trigger angle, rad, from squeezed to released (abc)."""
FOLLOWER_GRIPPER_RANGE = (-0.1, 1.0)
"""What abc maps that onto: the follower's opening, past closed at the squeezed end."""

JOINT_MIN = np.deg2rad([-120.0, 0.0, 0.0, -75.0, -85.0, -115.0])
JOINT_MAX = np.deg2rad([120.0, 180.0, 180.0, 75.0, 85.0, 115.0])

TORQUE_TO_CURRENT = 1158.73
"""XL330-M077 Goal Current units (mA) per N*m (abc)."""
CURRENT_LIMIT = 900
"""Goal Current units abc never exceeds."""
LIMIT_STIFFNESS = 0.3
"""N*m per rad past a soft joint limit, pushing back (abc)."""


def raw_angle(counts: npt.ArrayLike) -> FloatArray:
    """Encoder counts (4096 a turn, multi-turn) to rad, 2048 counts being zero (abc)."""
    return (np.asarray(counts, dtype=np.float64) / 2048 - 1) * np.pi


def calibrate(raw: npt.ArrayLike, signs: npt.ArrayLike) -> FloatArray:
    """The offsets that put each motor nearest its calibration position (abc).

    Each offset is a whole number of ``CALIBRATION_STEP``: if the GELLO was more than half a
    step (22.5 degrees) away from ``CALIBRATION_POSITION`` when it opened, that joint is off
    by a whole step, and stays so until the next open.
    """
    raw = np.asarray(raw, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    candidates = np.arange(-80, 81) * CALIBRATION_STEP
    offsets = np.empty(MOTORS)
    for index in range(MOTORS):
        positions = signs[index] * (raw[index] - candidates)
        offsets[index] = candidates[np.argmin(np.abs(positions - CALIBRATION_POSITION[index]))]
    return offsets


def joint_angles(raw: npt.ArrayLike, offsets: npt.ArrayLike, signs: npt.ArrayLike) -> FloatArray:
    """All seven motors in the follower's convention; the last is the trigger's angle."""
    return np.asarray(signs, dtype=np.float64) * (
        np.asarray(raw, dtype=np.float64) - np.asarray(offsets, dtype=np.float64)
    )


def follower_opening(trigger_angle: float) -> float:
    """abc's follower gripper command for a trigger angle: -0.1 (past closed) to 1 (open)."""
    return float(np.interp(trigger_angle, LEADER_GRIPPER_RANGE, FOLLOWER_GRIPPER_RANGE))


def trigger(trigger_angle: float) -> float:
    """xpisode's trigger, 0 released to 1 squeezed: ``1 - opening``, clipped to [0, 1]."""
    return float(np.clip(1.0 - follower_opening(trigger_angle), 0.0, 1.0))


def soft_limits() -> tuple[FloatArray, FloatArray]:
    """abc's: 1% inside each limit, and the elbow's top 30% further in."""
    low, high = JOINT_MIN.copy(), JOINT_MAX.copy()
    span = high - low
    low += 0.01 * span
    high -= 0.01 * span
    high[2] -= 0.3 * span[2]
    return low, high


def limit_torques(joints: FloatArray, signs: npt.ArrayLike) -> FloatArray:
    """N*m on each arm motor pushing a joint back inside its soft limits; zero inside."""
    low, high = soft_limits()
    excess = np.maximum(joints - high, 0.0) + np.minimum(joints - low, 0.0)
    return -LIMIT_STIFFNESS * excess * np.asarray(signs, dtype=np.float64)[:DOF]


def trigger_spring(opening: float, opening_rate: float) -> float:
    """N*m on the trigger toward released: abc's spring, capped at 0.03, with its damping.

    ``opening`` is ``follower_opening`` of the trigger angle, ``opening_rate`` its rate in 1/s.
    """
    return min(0.03, 0.05 * (1.5 - opening)) - 0.005 * opening_rate


def currents(torques: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """N*m per motor to Goal Current units, within abc's limit."""
    scaled = np.asarray(torques, dtype=np.float64) * TORQUE_TO_CURRENT
    return np.clip(np.rint(scaled), -CURRENT_LIMIT, CURRENT_LIMIT).astype(np.int64)
