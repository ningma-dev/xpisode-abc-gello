"""The GELLO's numbers: encoder counts to YAM joint angles, the trigger, and the currents of
its force feedback. Pure functions, ported from abc's ``deploy/robot/leaders/gello_leader.py``
(amazon-far/abc d0832d1, Apache-2.0), so they can be tested without the arm.

Joint angles are in the YAM follower's convention (i2rt's), as abc's leader publishes them.
The trigger is xpisode's leader convention, as YamLeader reports it: 0 released, 1 squeezed;
a teleop client maps it to the follower's opening as ``1 - trigger``. Between those, the
mapping is abc's: the follower opening it would have commanded is ``1 - trigger``.

Unlike abc, the zero, each joint's direction and the trigger's ends are measured once per
GELLO (``calibrate``) and kept in a file (``Calibration``): abc assumes one build's directions
and finds the zero again at every start, which needs the GELLO held in its zero pose then.
Here the encoders are absolute within a turn, so at every start each joint takes the whole
turn that puts it nearest the middle of its range: any pose will do.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DOF = 6
"""Arm joints; the seventh motor is the trigger."""
MOTORS = DOF + 1
TURN = 2 * np.pi

LEADER_GRIPPER_RANGE = (-0.6040225, 0.33736414)
"""Trigger angle, rad, from squeezed to released (abc)."""
FOLLOWER_GRIPPER_RANGE = (-0.1, 1.0)
"""What abc maps that onto: the follower's opening, past closed at the squeezed end."""

JOINT_MIN = np.deg2rad([-120.0, 0.0, 0.0, -75.0, -85.0, -115.0])
JOINT_MAX = np.deg2rad([120.0, 180.0, 180.0, 75.0, 85.0, 115.0])
JOINT_MIDDLE = (JOINT_MIN + JOINT_MAX) / 2
"""Each joint is read as the turn nearest this: every range is well under a turn wide."""

TORQUE_TO_CURRENT = 1158.73
"""XL330-M077 Goal Current units (mA) per N*m (abc)."""
CURRENT_LIMIT = 900
"""Goal Current units abc never exceeds."""
LIMIT_STIFFNESS = 0.3
"""N*m per rad past a soft joint limit, pushing back (abc)."""

FORMAT = "xpisode_abc_gello.calibration/1"


@dataclass(frozen=True)
class Calibration:
    """One GELLO, measured by ``xpisode_abc_gello.calibrate``. Angles are raw motor angles
    (``raw_angle``) as read then."""

    serial: str | None
    """The U2D2's FTDI serial: the file belongs to that GELLO."""
    servo_ids: tuple[int, ...]
    """M0 (base) .. M5 (wrist roll), then the trigger."""
    signs: tuple[int, ...]
    """Per arm joint, 1 if the motor turns with the YAM joint's positive direction, else -1."""
    zero: tuple[float, ...]
    """Per arm joint, the motor's angle with the GELLO in the YAM's zero pose."""
    trigger_released: float
    trigger_squeezed: float
    at: str
    """When it was measured (ISO 8601)."""

    def __post_init__(self) -> None:
        if len(self.servo_ids) != MOTORS or len(self.signs) != DOF or len(self.zero) != DOF:
            raise ValueError(f"a calibration has {MOTORS} servo_ids, {DOF} signs and zeros")
        if any(s not in (1, -1) for s in self.signs):
            raise ValueError("signs are 1 or -1")
        if abs(self.trigger_squeezed - self.trigger_released) < np.deg2rad(10):
            raise ValueError("the trigger's ends are less than 10 degrees apart")

    @property
    def sign_array(self) -> FloatArray:
        return np.asarray(self.signs, dtype=np.float64)

    @property
    def trigger_gain(self) -> float:
        """abc's trigger angle per rad of the motor: maps released and squeezed onto
        ``LEADER_GRIPPER_RANGE``. Its sign is the trigger motor's direction."""
        squeezed, released = LEADER_GRIPPER_RANGE
        return (squeezed - released) / (self.trigger_squeezed - self.trigger_released)

    @classmethod
    def load(cls, path: Path) -> Calibration:
        data = json.loads(Path(path).read_text())
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields["servo_ids"] = tuple(int(i) for i in fields["servo_ids"])
        fields["signs"] = tuple(int(s) for s in fields["signs"])
        fields["zero"] = tuple(float(z) for z in fields["zero"])
        return cls(**fields)

    def to_json(self, **extra: object) -> str:
        return json.dumps({"format": FORMAT, **asdict(self), **extra}, indent=1) + "\n"


def raw_angle(counts: npt.ArrayLike) -> FloatArray:
    """Encoder counts (4096 a turn, multi-turn) to rad, 2048 counts being zero (abc)."""
    return (np.asarray(counts, dtype=np.float64) / 2048 - 1) * np.pi


@dataclass(frozen=True)
class Offsets:
    """Where each motor's zero is in this session's raw angles: the calibration's, moved by the
    whole turns that put the GELLO's pose at the start nearest the middle of its ranges."""

    arm: FloatArray
    trigger: float


def offsets(calibration: Calibration, raw: npt.ArrayLike) -> Offsets:
    """For this session, from one reading of all seven motors."""
    raw = np.asarray(raw, dtype=np.float64)
    signs, zero = calibration.sign_array, np.asarray(calibration.zero, dtype=np.float64)
    # sign * (raw - offset) nearest the middle, offset = zero + whole turns
    turns = np.round((raw[:DOF] - signs * JOINT_MIDDLE - zero) / TURN)
    arm = zero + turns * TURN
    gain = calibration.trigger_gain
    middle = (sum(LEADER_GRIPPER_RANGE) / 2 - LEADER_GRIPPER_RANGE[1]) / gain
    turn = np.round((raw[DOF] - middle - calibration.trigger_released) / TURN)
    return Offsets(arm, float(calibration.trigger_released + turn * TURN))


def joint_angles(raw: npt.ArrayLike, offsets: Offsets, signs: npt.ArrayLike) -> FloatArray:
    """The six arm joints in the follower's convention, from all seven raw angles."""
    raw = np.asarray(raw, dtype=np.float64)
    return np.asarray(signs, dtype=np.float64) * (raw[:DOF] - offsets.arm)


def trigger_angle(raw: npt.ArrayLike, offsets: Offsets, gain: float) -> float:
    """abc's trigger angle (``LEADER_GRIPPER_RANGE``) from all seven raw angles."""
    return float(LEADER_GRIPPER_RANGE[1] + (np.asarray(raw, dtype=np.float64)[DOF] - offsets.trigger) * gain)


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
