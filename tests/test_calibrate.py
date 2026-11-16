"""The calibration procedure on a simulated GELLO, and the device reading through its file."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from xpisode_device import MonotonicClock

from xpisode_abc_gello import bus as bus_module
from xpisode_abc_gello import calibrate, device, mapping

IDS = (10, 11, 12, 13, 14, 15, 16)
SIGNS = np.array([1, -1, -1, 1, 1, -1])
ZERO = np.array([0.4, -1.1, 2.0, -0.3, 0.9, 2.9])
RELEASED, SQUEEZED = 0.5, -0.45


class Gello:
    """Raw motor angles of a GELLO with ``SIGNS`` and ``ZERO``, posed by the test."""

    def __init__(self) -> None:
        self.pose = np.zeros(6)
        self.trigger = RELEASED

    def raw(self) -> np.ndarray:
        return np.r_[ZERO + SIGNS * self.pose, self.trigger]


def scripted(gello: Gello, steps: list) -> calibrate.mapping.Calibration:
    """Run the procedure; each prompt applies the next step to the GELLO."""
    said: list[str] = []
    queue = list(steps)

    def ask(prompt: str) -> None:
        queue.pop(0)(gello)

    calibration = calibrate.run(gello.raw, ask, said.append, "FTBPFNFF", IDS)
    assert not queue
    scripted.said = said  # type: ignore[attr-defined]
    return calibration


def at_zero(g: Gello) -> None:
    g.pose, g.trigger = np.zeros(6), RELEASED


def turn(joint: int, degrees: float = 45.0):
    def apply(g: Gello) -> None:
        g.pose = np.zeros(6)
        g.pose[joint] = np.deg2rad(degrees)

    return apply


def squeeze(g: Gello) -> None:
    g.pose, g.trigger = np.zeros(6), SQUEEZED


def full_procedure() -> list:
    steps = [at_zero]
    for joint in range(6):
        steps += [turn(joint), at_zero]
    return [*steps, squeeze, at_zero]


def test_the_procedure_measures_signs_zero_and_trigger() -> None:
    calibration = scripted(Gello(), full_procedure())
    assert calibration.signs == tuple(SIGNS)
    np.testing.assert_allclose(calibration.zero, ZERO)
    assert calibration.trigger_released == pytest.approx(RELEASED)
    assert calibration.trigger_squeezed == pytest.approx(SQUEEZED)
    assert calibration.servo_ids == IDS and calibration.serial == "FTBPFNFF"


def test_a_failed_step_is_asked_again() -> None:
    steps = full_procedure()
    # joint 3's step: first too small a move, then the right one
    index = 1 + 2 * 2
    steps[index:index] = [turn(2, 5.0)]
    calibration = scripted(Gello(), steps)
    assert calibration.signs == tuple(SIGNS)
    assert any("joint 3 moved 5 deg" in line for line in scripted.said)  # type: ignore[attr-defined]


def test_swapped_motors_are_named() -> None:
    zero = np.zeros(7)
    moved = zero.copy()
    moved[3] = np.deg2rad(40)  # asked for joint 2, joint 4's motor turned
    with pytest.raises(calibrate.CalibrationError, match="joint 4 moved most"):
        calibrate.joint_sign(zero, moved, 1)


def test_another_joint_moving_along_is_refused() -> None:
    zero = np.zeros(7)
    moved = zero.copy()
    moved[1], moved[2] = np.deg2rad(45), np.deg2rad(-30)
    with pytest.raises(calibrate.CalibrationError, match="joint 3 moved too"):
        calibrate.joint_sign(zero, moved, 1)


def test_signs_either_way() -> None:
    zero = np.zeros(7)
    moved = zero.copy()
    moved[4] = np.deg2rad(-50)
    assert calibrate.joint_sign(zero, moved, 4) == -1
    moved[4] = np.deg2rad(50)
    assert calibrate.joint_sign(zero, moved, 4) == 1


# --- the device, on a simulated bus ----------------------------------------------------------


class FakeBus:
    gello = Gello()
    turns = np.zeros(7)
    writes: ClassVar[list[np.ndarray]] = []
    mode_changes = 0

    def __init__(self, device: str, ids: tuple[int, ...], baudrate: int) -> None:
        self.port = self
        FakeBus.writes = []

    def closePort(self) -> None:
        pass

    def ping_all(self) -> dict[int, int]:
        return dict.fromkeys(IDS, 1190)

    def torque_off(self) -> None:
        pass

    def enter_current_mode(self) -> None:
        FakeBus.mode_changes += 1

    def read(self) -> bus_module.Present:
        raw = self.gello.raw() + self.turns
        counts = np.rint((raw / np.pi + 1) * 2048).astype(np.int64)
        return bus_module.Present(counts, np.zeros(7), np.zeros(7, dtype=np.int64))

    def write_currents(self, currents: np.ndarray, *, force: bool = False) -> None:
        FakeBus.writes.append(np.asarray(currents))

    def close(self) -> None:
        pass


@pytest.fixture
def calibration_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(device, "Bus", FakeBus)
    monkeypatch.setattr(device, "latency_timer", lambda _: 1)
    calibration = scripted(Gello(), full_procedure())
    path = tmp_path / "gello_leader_l.json"
    path.write_text(calibration.to_json())
    return path


PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBPFNFF-if00-port0"


def test_the_device_opens_in_any_pose_and_reads_the_yam_convention(calibration_file: Path) -> None:
    FakeBus.gello = Gello()
    FakeBus.gello.pose = np.deg2rad([30.0, 100.0, 40.0, -20.0, 10.0, 60.0])
    FakeBus.turns = np.array([1, -2, 0, 3, 0, -1, 2]) * mapping.TURN
    gello = device.GelloLeader(PORT, IDS, str(calibration_file), force_feedback=False)
    gello.open(MonotonicClock())
    try:
        reading = gello.read("input")
        assert reading is not None
        np.testing.assert_allclose(np.rad2deg(reading.values[0]), [30, 100, 40, -20, 10, 60],
                                   atol=0.2)
        assert float(reading.values[1][0]) == pytest.approx(0.0, abs=0.01)
    finally:
        gello.close()


def test_the_trigger_spring_pushes_toward_released_on_the_motor(calibration_file: Path) -> None:
    FakeBus.gello = Gello()
    FakeBus.turns = np.zeros(7)
    FakeBus.gello.trigger = (RELEASED + SQUEEZED) / 2  # half squeezed
    gello = device.GelloLeader(PORT, IDS, str(calibration_file))
    gello.open(MonotonicClock())
    gello.close()
    trigger_current = FakeBus.writes[-1][6]
    # released is the larger raw angle here, so toward released is a positive motor current
    assert trigger_current > 0


def test_a_calibration_of_another_gello_is_refused(calibration_file: Path) -> None:
    other = PORT.replace("FTBPFNFF", "FTAAAAAA")
    with pytest.raises(ValueError, match="U2D2 FTBPFNFF"):
        device.GelloLeader(other, IDS, str(calibration_file))
    with pytest.raises(ValueError, match="servo IDs"):
        device.GelloLeader(PORT, (20, 21, 22, 23, 24, 25, 26), str(calibration_file))
