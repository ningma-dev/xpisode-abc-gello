"""The ported numbers against abc's own formulas, and the device's calibration guard."""

from __future__ import annotations

import numpy as np
import pytest

from xpisode_abc_gello import mapping

SIGNS = np.array([1, -1, -1, -1, 1, 1, 1], dtype=np.float64)


def abc_calibrate(raw: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """abc's GelloLeaderNode._calibrate_offsets, verbatim but for self."""
    candidates = np.arange(-80, 81) * (np.pi / 4)
    offsets = []
    for index, value in enumerate(raw):
        positions = signs[index] * (value - candidates)
        error = np.abs(positions - mapping.CALIBRATION_POSITION[index])
        offsets.append(candidates[np.argmin(error)])
    return np.asarray(offsets)


def abc_read_position(raw: np.ndarray, offsets: np.ndarray, signs: np.ndarray) -> np.ndarray:
    position = signs * (raw - offsets)
    position[-1] = np.interp(
        position[-1], mapping.LEADER_GRIPPER_RANGE, mapping.FOLLOWER_GRIPPER_RANGE
    )
    return position


def test_raw_angle_matches_abc() -> None:
    counts = np.array([0, 1024, 2048, 3072, 4096, -2048, 8192])
    np.testing.assert_allclose(mapping.raw_angle(counts), (counts / 2048 - 1) * np.pi)


@pytest.mark.parametrize("seed", range(20))
def test_calibration_and_joints_match_abc(seed: int) -> None:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-20, 20, size=7)
    offsets = mapping.calibrate(raw, SIGNS)
    np.testing.assert_allclose(offsets, abc_calibrate(raw, SIGNS))
    later = raw + rng.uniform(-1, 1, size=7)
    expected = abc_read_position(later, offsets, SIGNS)
    angles = mapping.joint_angles(later, offsets, SIGNS)
    np.testing.assert_allclose(angles[:6], expected[:6])
    opening = mapping.follower_opening(float(angles[-1]))
    assert opening == pytest.approx(expected[-1])
    assert mapping.trigger(float(angles[-1])) == pytest.approx(np.clip(1 - expected[-1], 0, 1))


def test_calibration_snaps_within_half_a_step() -> None:
    near = mapping.CALIBRATION_POSITION + np.deg2rad(10)
    offsets = mapping.calibrate(near, np.ones(7))
    pose = mapping.joint_angles(near, offsets, np.ones(7))
    np.testing.assert_allclose(pose - mapping.CALIBRATION_POSITION, np.deg2rad(10), atol=1e-9)


def test_trigger_ends() -> None:
    squeezed, released = mapping.LEADER_GRIPPER_RANGE
    assert mapping.trigger(released) == 0.0
    assert mapping.trigger(released + 0.2) == 0.0
    assert mapping.trigger(squeezed) == 1.0
    assert mapping.trigger(mapping.CALIBRATION_POSITION[-1]) == 0.0


def test_limit_torques_push_back_and_vanish_inside() -> None:
    low, high = mapping.soft_limits()
    np.testing.assert_array_equal(mapping.limit_torques((low + high) / 2, SIGNS), np.zeros(6))
    past = high + 0.1
    torques = mapping.limit_torques(past, SIGNS)
    np.testing.assert_allclose(torques, -0.3 * 0.1 * SIGNS[:6])


def test_trigger_spring_is_abcs() -> None:
    assert mapping.trigger_spring(1.0, 0.0) == pytest.approx(0.025)
    assert mapping.trigger_spring(0.0, 0.0) == pytest.approx(0.03)
    assert mapping.trigger_spring(0.5, 2.0) == pytest.approx(0.03 - 0.01)


def test_currents_are_abcs_units_and_limit() -> None:
    np.testing.assert_array_equal(mapping.currents([0.03, -0.03, 2.0]), [35, -35, 900])
