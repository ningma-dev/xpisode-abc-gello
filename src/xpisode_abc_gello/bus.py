"""The seven XL330s on one U2D2: one sync read and one sync write a cycle.

Ported from abc's ``deploy/robot/leaders/dynamixel.py`` and ``leader_robot.py`` (amazon-far/abc
d0832d1, Apache-2.0). Differences: present current, velocity and position come in one sync read,
and the seven goal currents go out in one sync write instead of seven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from dynamixel_sdk import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_LOBYTE,
    GroupSyncRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)

OPERATING_MODE = 11
TORQUE_ENABLE = 64
HARDWARE_ERROR = 70
GOAL_CURRENT = 102
PRESENT_CURRENT = 126  # 2 bytes, then PRESENT_VELOCITY 128 (4), then PRESENT_POSITION 132 (4)
PRESENT_BLOCK = 10
CURRENT_MODE = 0
VELOCITY_UNIT = 0.229 * 2 * np.pi / 60
"""rad/s per Present Velocity unit."""
VOLTAGE_WARNING = 128
"""The Packet's error byte with only the alert bit: abc ignores it, as the old driver did."""


def _signed(value: int, nbytes: int) -> int:
    bits = 8 * nbytes
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


@dataclass(frozen=True)
class Present:
    counts: npt.NDArray[np.int64]
    velocity: npt.NDArray[np.float64]
    """rad/s, motor sign."""
    current: npt.NDArray[np.int64]


def latency_timer(device: str) -> int:
    """The FTDI driver's latency timer behind ``device``: it must be 1 ms (abc refuses else)."""
    name = Path(device).resolve().name
    return int((Path("/sys/bus/usb-serial/devices") / name / "latency_timer").read_text().strip())


class _Port(PortHandler):
    """The SDK sizes a reply's timeout for the FTDI default latency of 16 ms (34 ms in all);
    the device insists on 1 ms, so a lost reply costs a few ms here, not a dozen cycles."""

    LATENCY_MS = 1.0

    def setPacketTimeout(self, packet_length: int) -> None:
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = self.tx_time_per_byte * packet_length + self.LATENCY_MS * 2.0 + 2.0


class Bus:
    def __init__(self, device: str, ids: tuple[int, ...], baudrate: int) -> None:
        self.ids = ids
        self.port = _Port(device)
        self.packets = PacketHandler(2.0)
        if not self.port.openPort():
            raise ConnectionError(f"cannot open {device}")
        if not self.port.setBaudRate(baudrate):
            self.port.closePort()
            raise ConnectionError(f"cannot set {device} to {baudrate} baud")
        self._reader = GroupSyncRead(self.port, self.packets, PRESENT_CURRENT, PRESENT_BLOCK)
        for motor in ids:
            if not self._reader.addParam(motor):
                raise RuntimeError(f"cannot register XL330 {motor}")
        self._writer = GroupSyncWrite(self.port, self.packets, GOAL_CURRENT, 2)
        self.current_mode = False

    def _check(self, result: int, error: int, motor: int, what: str) -> None:
        if result != COMM_SUCCESS:
            raise ConnectionError(f"XL330 {motor}: {what}: {self.packets.getTxRxResult(result)}")
        if error not in (0, VOLTAGE_WARNING):
            raise ConnectionError(f"XL330 {motor}: {what}: error {error}")

    def write1(self, motor: int, address: int, value: int, what: str) -> None:
        result, error = self.packets.write1ByteTxRx(self.port, motor, address, value)
        self._check(result, error, motor, what)

    def read1(self, motor: int, address: int, what: str) -> int:
        value, result, error = self.packets.read1ByteTxRx(self.port, motor, address)
        self._check(result, error, motor, what)
        return int(value)

    def ping_all(self) -> dict[int, int]:
        """Each motor's model number; raises naming the first that does not answer."""
        models = {}
        for motor in self.ids:
            model, result, error = self.packets.ping(self.port, motor)
            self._check(result, error, motor, "ping")
            models[motor] = int(model)
        return models

    def torque_off(self) -> None:
        for motor in self.ids:
            self.write1(motor, TORQUE_ENABLE, 0, "torque off")
        self.current_mode = False

    def enter_current_mode(self) -> None:
        """Current control with zero current: the arm still floats, until currents come."""
        self.write_currents(np.zeros(len(self.ids), dtype=np.int64), force=True)
        for motor in self.ids:
            self.write1(motor, TORQUE_ENABLE, 0, "torque off")
            self.write1(motor, OPERATING_MODE, CURRENT_MODE, "current mode")
            self.write1(motor, TORQUE_ENABLE, 1, "torque on")
        self.current_mode = True

    def read(self) -> Present:
        result = self._reader.txRxPacket()
        if result != COMM_SUCCESS:
            raise ConnectionError(f"sync read: {self.packets.getTxRxResult(result)}")
        counts, velocity, current = [], [], []
        for motor in self.ids:
            if not self._reader.isAvailable(motor, PRESENT_CURRENT, PRESENT_BLOCK):
                raise ConnectionError(f"sync read: no reply from XL330 {motor}")
            current.append(_signed(self._reader.getData(motor, PRESENT_CURRENT, 2), 2))
            velocity.append(_signed(self._reader.getData(motor, PRESENT_CURRENT + 2, 4), 4))
            counts.append(_signed(self._reader.getData(motor, PRESENT_CURRENT + 6, 4), 4))
        return Present(
            np.array(counts, dtype=np.int64),
            np.array(velocity, dtype=np.float64) * VELOCITY_UNIT,
            np.array(current, dtype=np.int64),
        )

    def write_currents(self, currents: npt.NDArray[np.int64], *, force: bool = False) -> None:
        if not (self.current_mode or force):
            return
        for motor, value in zip(self.ids, currents, strict=True):
            raw = int(value) & 0xFFFF
            if not self._writer.addParam(motor, [DXL_LOBYTE(raw), DXL_HIBYTE(raw)]):
                raise RuntimeError(f"cannot command XL330 {motor}")
        result = self._writer.txPacket()
        self._writer.clearParam()
        if result != COMM_SUCCESS:
            raise ConnectionError(f"sync write: {self.packets.getTxRxResult(result)}")

    def close(self) -> None:
        try:
            self.torque_off()
        finally:
            self.port.closePort()
