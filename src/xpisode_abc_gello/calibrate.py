"""Measure a GELLO once, after it is built (or a horn is put back on): each joint's zero and
direction against the YAM's, and the trigger's ends. Writes the file GelloLeader reads.

    python -m xpisode_abc_gello.calibrate --device /dev/serial/by-id/...FTBPFNFF... \\
        --ids 10 11 12 13 14 15 16 --out tuned/gello_leader_l.json

The motors stay unpowered; nothing but the GELLO in the operator's hands moves. It asks for the
zero pose, then one joint at a time turned about 45 degrees the way the YAM's joint turns
positive (``DIRECTIONS``), back to zero each time, then the trigger squeezed. Each move must turn
that joint's motor and no other much, which also finds IDs on the wrong joints. It ends with
the zero pose again and shows how far each joint reads from zero, then writes the file.

The directions are the YAM's (i2rt's model): seen from behind the arm, looking the way the
gripper points in the zero pose (forward).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from xpisode_abc_gello import mapping
from xpisode_abc_gello.bus import Bus, latency_timer
from xpisode_abc_gello.device import _ftdi_serial

DIRECTIONS = (
    "joint 1 (base): turn the arm to the left (counterclockwise seen from above)",
    "joint 2 (shoulder): lift the upper arm",
    "joint 3 (elbow): lift the forearm, unfolding the elbow",
    "joint 4 (wrist pitch): tilt the gripper's tip up",
    "joint 5 (wrist yaw): turn the gripper's tip to the right (clockwise seen from above)",
    "joint 6 (wrist roll): roll the gripper clockwise as seen from behind (its left side up)",
)
"""How to turn each joint positive, from the zero pose, in the YAM's convention."""
MIN_MOVE = np.deg2rad(20)
"""A joint turned for its step must have moved at least this."""
MAX_OTHER = 0.5
"""No other arm motor may move more than this fraction of the joint's own move."""
ZERO_TOLERANCE = np.deg2rad(8)
"""Back at zero, every motor must be this close to where the zero pose put it."""


class CalibrationError(ValueError):
    pass


def joint_sign(zero: np.ndarray, moved: np.ndarray, joint: int) -> int:
    """The direction of ``joint`` from the raw angles at zero and after turning it positive."""
    delta = np.asarray(moved, dtype=np.float64)[: mapping.DOF] - np.asarray(zero)[: mapping.DOF]
    own = abs(delta[joint])
    biggest = int(np.argmax(np.abs(delta)))
    if biggest != joint and abs(delta[biggest]) >= MIN_MOVE:
        raise CalibrationError(
            f"joint {joint + 1} was asked for, but the motor of joint {biggest + 1} moved most "
            f"({np.rad2deg(delta[biggest]):+.0f} deg): two motors' IDs are swapped, or the "
            "wrong joint was turned"
        )
    if own < MIN_MOVE:
        raise CalibrationError(
            f"joint {joint + 1} moved {np.rad2deg(own):.0f} deg; turn it about 45"
        )
    others = np.delete(np.abs(delta), joint)
    if np.any(others > MAX_OTHER * own):
        worst = int(np.argmax(np.where(np.arange(mapping.DOF) == joint, 0, np.abs(delta))))
        raise CalibrationError(
            f"while joint {joint + 1} turned, joint {worst + 1} moved too "
            f"({np.rad2deg(delta[worst]):+.0f} deg); hold the others still"
        )
    return 1 if delta[joint] > 0 else -1


def check_back_at_zero(zero: np.ndarray, now: np.ndarray) -> None:
    away = np.abs(np.asarray(now)[: mapping.DOF] - np.asarray(zero)[: mapping.DOF])
    if np.any(away > ZERO_TOLERANCE):
        far = {f"joint {i + 1}": round(float(np.rad2deg(a))) for i, a in enumerate(away) if a > ZERO_TOLERANCE}
        raise CalibrationError(f"not back at the zero pose, degrees away: {far}")


def build(
    serial: str | None,
    ids: Sequence[int],
    zero: np.ndarray,
    signs: Sequence[int],
    squeezed: float,
    at: str,
) -> mapping.Calibration:
    return mapping.Calibration(
        serial=serial,
        servo_ids=tuple(int(i) for i in ids),
        signs=tuple(int(s) for s in signs),
        zero=tuple(float(z) for z in np.asarray(zero)[: mapping.DOF]),
        trigger_released=float(zero[mapping.DOF]),
        trigger_squeezed=float(squeezed),
        at=at,
    )


def run(read: Callable[[], np.ndarray], ask: Callable[[str], None], say: Callable[[str], None],
        serial: str | None, ids: Sequence[int]) -> mapping.Calibration:
    """The procedure, on ``read`` (raw angles of the seven motors, averaged) and ``ask`` (show a
    step, wait until the operator is done). A failed step is asked again."""

    def step(prompt: str, check: Callable[[np.ndarray], object]) -> object:
        while True:
            ask(prompt)
            now = read()
            try:
                return check(now)
            except CalibrationError as error:
                say(f"  {error}. Again.")

    ask("Put the GELLO in the YAM's zero pose (folded, gripper pointing forward), trigger released.")
    zero = read()
    signs = []
    for joint, direction in enumerate(DIRECTIONS):
        sign = step(f"From zero, {direction}, about 45 degrees, and hold it.",
                    lambda now, j=joint: joint_sign(zero, now, j))
        say(f"  joint {joint + 1}: motor {ids[joint]}, direction {sign:+d}")
        signs.append(sign)
        step("Back to the zero pose.", lambda now: check_back_at_zero(zero, now))

    def squeezed_trigger(now: np.ndarray) -> float:
        if abs(now[mapping.DOF] - zero[mapping.DOF]) < np.deg2rad(10):
            raise CalibrationError("the trigger hardly moved; squeeze it all the way")
        check_back_at_zero(zero, now)
        return float(now[mapping.DOF])

    squeezed = step("Squeeze the trigger all the way, the arm still at zero.", squeezed_trigger)
    at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    calibration = build(serial, ids, zero, signs, squeezed, at)

    def verify(now: np.ndarray) -> None:
        offsets = mapping.offsets(calibration, now)
        pose = np.rad2deg(mapping.joint_angles(now, offsets, calibration.sign_array))
        trigger = mapping.trigger(mapping.trigger_angle(now, offsets, calibration.trigger_gain))
        say(f"  joints read {np.array2string(pose, precision=1, suppress_small=True)} deg, "
            f"trigger {trigger:.2f}")
        check_back_at_zero(zero, now)

    step("Release the trigger, back to the zero pose: a last check.", verify)
    return calibration


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xpisode_abc_gello.calibrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ids", type=int, nargs=mapping.MOTORS, required=True)
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    timer = latency_timer(args.device)
    if timer != 1:
        sys.exit(f"the FTDI latency timer of {args.device} is {timer} ms; set it to 1")
    bus = Bus(args.device, tuple(args.ids), args.baudrate)
    try:
        bus.ping_all()
        bus.torque_off()

        def read() -> np.ndarray:
            return np.mean([mapping.raw_angle(bus.read().counts) for _ in range(20)], axis=0)

        def ask(prompt: str) -> None:
            input(f"{prompt}\n  Enter when done: ")

        calibration = run(read, ask, print, _ftdi_serial(args.device), args.ids)
    finally:
        bus.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(calibration.to_json(by="xpisode_abc_gello.calibrate"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
