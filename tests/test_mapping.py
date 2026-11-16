"""The ported numbers against abc's own formulas, and the calibration that replaces abc's
start-up zeroing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xpisode_abc_gello import mapping

ABC_SIGNS = np.array([1, -1, -1, -1, 1, 1, 1], dtype=np.float64)


def abc_read_position(raw: np.ndarray, offsets: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """abc's GelloLeaderNode read: signs * (raw - offsets), the trigger interpolated."""
    position = signs * (raw - offsets)
    position[-1] = np.interp(
        position[-1], mapping.LEADER_GRIPPER_RANGE, mapping.FOLLOWER_GRIPPER_RANGE
    )
    return position


def calibration_like_abc(abc_offsets: np.ndarray, signs: np.ndarray) -> mapping.Calibration:
    """The file that reads as abc does with these offsets: the zero pose is raw == offset, and
    the trigger's ends are its offset plus abc's released and squeezed angles."""
    squeezed, released = mapping.LEADER_GRIPPER_RANGE
    return mapping.Calibration(
        serial=None,
        servo_ids=tuple(range(10, 17)),
        signs=tuple(int(s) for s in signs[:6]),
        zero=tuple(abc_offsets[:6]),
        trigger_released=float(abc_offsets[6] + released),
        trigger_squeezed=float(abc_offsets[6] + squeezed),
        at="2026-11-16T00:00:00+00:00",
    )


def random_pose(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(mapping.JOINT_MIN, mapping.JOINT_MAX)


def test_raw_angle_matches_abc() -> None:
    counts = np.array([0, 1024, 2048, 3072, 4096, -2048, 8192])
    np.testing.assert_allclose(mapping.raw_angle(counts), (counts / 2048 - 1) * np.pi)


@pytest.mark.parametrize("seed", range(20))
def test_reads_as_abc_with_abcs_offsets(seed: int) -> None:
    rng = np.random.default_rng(seed)
    abc_offsets = rng.integers(-8, 9, size=7) * (np.pi / 4)
    calibration = calibration_like_abc(abc_offsets, ABC_SIGNS)
    pose = random_pose(rng)
    trigger_angle = rng.uniform(*mapping.LEADER_GRIPPER_RANGE)
    raw = abc_offsets + ABC_SIGNS * np.r_[pose, trigger_angle]
    offsets = mapping.offsets(calibration, raw)
    expected = abc_read_position(raw, abc_offsets, ABC_SIGNS)
    np.testing.assert_allclose(mapping.joint_angles(raw, offsets, ABC_SIGNS[:6]), expected[:6])
    angle = mapping.trigger_angle(raw, offsets, calibration.trigger_gain)
    assert mapping.follower_opening(angle) == pytest.approx(expected[-1])


@pytest.mark.parametrize("seed", range(30))
def test_any_pose_opens_whatever_turn_the_encoders_are_on(seed: int) -> None:
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=6)
    zero = rng.uniform(-np.pi, np.pi, size=6)
    calibration = mapping.Calibration(
        serial="X", servo_ids=tuple(range(7)), signs=tuple(int(s) for s in signs),
        zero=tuple(zero), trigger_released=0.3, trigger_squeezed=-0.6, at="t",
    )
    pose = random_pose(rng)
    turns = rng.integers(-3, 4, size=7) * mapping.TURN
    raw = np.r_[zero + signs * pose, 0.3] + turns
    offsets = mapping.offsets(calibration, raw)
    np.testing.assert_allclose(mapping.joint_angles(raw, offsets, signs), pose, atol=1e-9)
    released = mapping.trigger(mapping.trigger_angle(raw, offsets, calibration.trigger_gain))
    assert released == 0.0


def test_a_reversed_trigger_reads_the_same() -> None:
    forward = mapping.Calibration(None, tuple(range(7)), (1,) * 6, (0.0,) * 6, 0.3, -0.6, "t")
    backward = mapping.Calibration(None, tuple(range(7)), (1,) * 6, (0.0,) * 6, 0.3, 1.2, "t")
    assert forward.trigger_gain > 0 > backward.trigger_gain
    for fraction in (0.0, 0.25, 0.5, 1.0):
        values = []
        for calibration in (forward, backward):
            raw = np.zeros(7)
            raw[6] = calibration.trigger_released + fraction * (
                calibration.trigger_squeezed - calibration.trigger_released
            )
            offsets = mapping.offsets(calibration, raw)
            angle = mapping.trigger_angle(raw, offsets, calibration.trigger_gain)
            values.append(mapping.trigger(angle))
        assert values[0] == pytest.approx(values[1])
        # abc's: squeezed fully is an opening of -0.1, so the trigger reaches 1 a little early
        expected = np.clip(1 - np.interp(fraction, [0, 1], mapping.FOLLOWER_GRIPPER_RANGE[::-1]), 0, 1)
        assert values[0] == pytest.approx(expected, abs=1e-9)


def test_calibration_file_round_trip_and_checks(tmp_path: Path) -> None:
    calibration = mapping.Calibration("FTBPFNFF", tuple(range(10, 17)), (1, -1, -1, 1, 1, 1),
                                      (0.1,) * 6, 0.3, -0.6, "t")
    path = tmp_path / "gello.json"
    path.write_text(calibration.to_json(by="test"))
    assert mapping.Calibration.load(path) == calibration
    with pytest.raises(ValueError, match="signs"):
        mapping.Calibration(None, tuple(range(7)), (1, 0, 1, 1, 1, 1), (0.0,) * 6, 0.3, -0.6, "t")
    with pytest.raises(ValueError, match="trigger"):
        mapping.Calibration(None, tuple(range(7)), (1,) * 6, (0.0,) * 6, 0.3, 0.35, "t")
    path.write_text('{"format": "other"}')
    with pytest.raises(ValueError, match="not a"):
        mapping.Calibration.load(path)


def test_trigger_ends() -> None:
    squeezed, released = mapping.LEADER_GRIPPER_RANGE
    assert mapping.trigger(released) == 0.0
    assert mapping.trigger(released + 0.2) == 0.0
    assert mapping.trigger(squeezed) == 1.0


def test_limit_torques_push_back_and_vanish_inside() -> None:
    low, high = mapping.soft_limits()
    np.testing.assert_array_equal(mapping.limit_torques((low + high) / 2, ABC_SIGNS), np.zeros(6))
    past = high + 0.1
    torques = mapping.limit_torques(past, ABC_SIGNS)
    np.testing.assert_allclose(torques, -0.3 * 0.1 * ABC_SIGNS[:6])


def test_trigger_spring_is_abcs() -> None:
    assert mapping.trigger_spring(1.0, 0.0) == pytest.approx(0.025)
    assert mapping.trigger_spring(0.0, 0.0) == pytest.approx(0.03)
    assert mapping.trigger_spring(0.5, 2.0) == pytest.approx(0.03 - 0.01)


def test_currents_are_abcs_units_and_limit() -> None:
    np.testing.assert_array_equal(mapping.currents([0.03, -0.03, 2.0]), [35, -35, 900])
