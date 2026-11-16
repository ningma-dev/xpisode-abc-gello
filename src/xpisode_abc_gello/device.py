"""abc's GELLO as an xpisode leader, read like YamLeader: ``input`` with ``joint_pos`` and
``trigger``. A teleop client written for YamLeader drives a follower from it unchanged.

One thread owns the bus at ``rate_hz``: a sync read, then (with ``force_feedback``) a sync write
of the currents: abc's push-back at the soft joint limits, abc's trigger spring, and the
``trigger_feedback`` torque a client sends, for example from the follower's gripper effort.
``read`` hands out the newest cycle. The GELLO cannot be driven to a pose, so it has no
joint_position endpoint and a teleop client gives it no ``mirror``.

    xpisode-device run --station lab --name leader_l --factory xpisode_abc_gello:GelloLeader \\
        --config '{"device": "/dev/serial/by-id/usb-FTDI_..._FTBPFNFF-if00-port0",
                   "servo_ids": [10, 11, 12, 13, 14, 15, 16],
                   "calibration": "tuned/gello_leader_l.json"}'

The calibration (``python -m xpisode_abc_gello.calibrate``) gives each joint's zero and direction
and the trigger's ends; it must be this GELLO's (its U2D2 serial and servo IDs). Any pose opens.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xpisode_device import (
    Clock,
    CommandRejectedError,
    Device,
    DeviceInfo,
    Endpoint,
    Limits,
    Reading,
    Stream,
    Values,
)
from xpisode_format.model import JsonValue
from xpisode_format.spec import Axis, Field, RecordStream, Role, TimeMeaning

from xpisode_abc_gello import mapping
from xpisode_abc_gello.bus import Bus, latency_timer

ABC_SOURCE = "amazon-far/abc d0832d1 deploy/robot/leaders"
FAILURES_TO_FAULT = 5
"""Consecutive failed cycles before the device faults; one alone is retried."""
OPENING_SLOPE = (mapping.FOLLOWER_GRIPPER_RANGE[1] - mapping.FOLLOWER_GRIPPER_RANGE[0]) / (
    mapping.LEADER_GRIPPER_RANGE[1] - mapping.LEADER_GRIPPER_RANGE[0]
)


@dataclass(frozen=True)
class _Cycle:
    time_ns: int
    joints: np.ndarray
    trigger: float
    trigger_torque: float
    feedback_torque: float


def _scalar(name: str, unit: str, description: str) -> Field:
    return Field(name=name, dtype="float64", shape=(1,), unit=unit, description=description)


class GelloLeader(Device):
    driver = "xpisode_abc_gello.gello_leader"

    def __init__(
        self,
        device: str,
        servo_ids: Sequence[int],
        calibration: str,
        *,
        baudrate: int = 2_000_000,
        rate_hz: float = 400.0,
        poll_hz: float = 200.0,
        force_feedback: bool = True,
        max_feedback_torque: float = 0.15,
        feedback_timeout_s: float = 0.1,
        feedback_slew: float = 5.0,
    ) -> None:
        if len(servo_ids) != mapping.MOTORS:
            raise ValueError(f"servo_ids needs {mapping.MOTORS} entries")
        self.device = device
        self.ids = tuple(int(i) for i in servo_ids)
        self.calibration_path = Path(calibration)
        self.calibration = cal = mapping.Calibration.load(self.calibration_path)
        if cal.servo_ids != self.ids:
            raise ValueError(
                f"{calibration} is for servo IDs {list(cal.servo_ids)}, not {list(self.ids)}"
            )
        serial = _ftdi_serial(device)
        if cal.serial is not None and serial is not None and cal.serial != serial:
            raise ValueError(f"{calibration} is the GELLO on U2D2 {cal.serial}, not {serial}")
        self.signs = cal.sign_array
        self.gain = cal.trigger_gain
        self.baudrate = baudrate
        self.rate_hz = rate_hz
        self.poll_hz = poll_hz
        self.force_feedback = force_feedback
        self.max_feedback_torque = max_feedback_torque
        self.feedback_timeout_s = feedback_timeout_s
        self.feedback_slew = feedback_slew
        self._clock: Clock | None = None
        self._bus: Bus | None = None
        self._offsets = mapping.Offsets(np.zeros(mapping.DOF), 0.0)
        self._facts: dict[str, JsonValue] = {}
        self._lock = threading.Lock()
        self._cycle: _Cycle | None = None
        self._served: _Cycle | None = None
        self._fault: str | None = None
        self._feedback = 0.0
        self._feedback_at = -np.inf
        self._applied_feedback = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats = {"cycles": 0, "overruns": 0, "retries": 0, "max_cycle_ms": 0.0}

    # --- lifecycle ---------------------------------------------------------------------------

    def open(self, clock: Clock) -> None:
        self._clock = clock
        timer = latency_timer(self.device)
        if timer != 1:
            raise RuntimeError(
                f"the FTDI latency timer of {self.device} is {timer} ms; set it to 1 "
                "(a udev rule: ATTR{latency_timer}=\"1\")"
            )
        self._bus = bus = Bus(self.device, self.ids, self.baudrate)
        try:
            models = bus.ping_all()
            bus.torque_off()
            if self.force_feedback:
                bus.enter_current_mode()
            # The turns are taken after any mode change, from what the cycles will read.
            for _ in range(10):
                present = bus.read()
            raw = mapping.raw_angle(present.counts)
            self._offsets = mapping.offsets(self.calibration, raw)
            pose = mapping.joint_angles(raw, self._offsets, self.signs)
        except BaseException:
            bus.port.closePort()
            self._bus = None
            raise
        self._facts = {
            "driver": self.driver,
            "port": str(Path(self.device).resolve()),
            "device": self.device,
            "serial": _ftdi_serial(self.device),
            "servo_ids": list(self.ids),
            "models": [models[i] for i in self.ids],
            "baudrate": self.baudrate,
            "rate_hz": self.rate_hz,
            "calibration": str(self.calibration_path),
            "calibrated_at": self.calibration.at,
            "joint_signs": [int(s) for s in self.signs],
            "trigger_gain": round(self.gain, 6),
            "opening_pose_deg": [round(float(np.rad2deg(a)), 1) for a in pose],
            "force_feedback": self.force_feedback,
            "max_feedback_torque": self.max_feedback_torque,
            "ported_from": ABC_SOURCE,
        }
        self._step()  # the first cycle, so the first read has one
        self._thread = threading.Thread(target=self._loop, name="gello-bus", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._bus is not None:
            self._bus.close()  # torque off: the GELLO floats, as it does unpowered
            self._bus = None

    # --- the protocol ------------------------------------------------------------------------

    def describe(self) -> DeviceInfo:
        joint = Field(
            name="joint_pos",
            dtype="float64",
            shape=(mapping.DOF,),
            unit="rad",
            axes=(Axis(name="joint", labels=tuple(f"joint{i + 1}" for i in range(mapping.DOF))),),
            description="the GELLO's joints in the YAM follower's convention",
        )
        fields = (
            joint,
            _scalar("trigger", "1", "0 released, 1 squeezed; abc's follower opening is 1 - this"),
            _scalar(
                "trigger_torque",
                "N*m",
                "applied on the trigger toward released: abc's spring plus feedback_torque",
            ),
            _scalar("feedback_torque", "N*m", "the part of trigger_torque a client asked for"),
        )
        spec = RecordStream(
            role=Role.INPUT,
            time_meaning=TimeMeaning.RECEIVE,
            fields=fields,
            nominal_hz=self.poll_hz,
            description=f"Stamped when the sync read's reply arrived; the bus runs at {self.rate_hz:g} Hz.",
        )
        feedback = _scalar(
            "trigger_torque", "N*m", "pushes the trigger toward released, on top of the spring"
        )
        endpoints = {}
        if self.force_feedback:
            endpoints["trigger_feedback"] = Endpoint(
                resource="trigger",
                kind="joint_torque",
                fields=(feedback,),
                limits={"trigger_torque": Limits(min=(0.0,), max=(self.max_feedback_torque,))},
                description=(
                    "Resistance the operator feels when squeezing, for the follower's grip; "
                    f"zero {self.feedback_timeout_s:g} s after the last target or when halted."
                ),
            )
        return DeviceInfo(
            device=self._facts,
            streams={"input": Stream(spec, poll_hz=self.poll_hz)},
            endpoints=endpoints,
        )

    def read(self, stream: str) -> Reading | None:
        with self._lock:
            fault, cycle = self._fault, self._cycle
            if fault is not None:
                raise RuntimeError(fault)
            if cycle is None or cycle is self._served:
                return None
            self._served = cycle
        return Reading(
            cycle.time_ns,
            (
                cycle.joints,
                np.array([cycle.trigger]),
                np.array([cycle.trigger_torque]),
                np.array([cycle.feedback_torque]),
            ),
        )

    def submit(self, endpoint: str, values: Values) -> None:
        torque = float(np.asarray(values[0], dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(torque):
            raise CommandRejectedError("trigger_torque must be finite")
        with self._lock:
            self._feedback = float(np.clip(torque, 0.0, self.max_feedback_torque))
            self._feedback_at = time.monotonic()

    def halt(self, resource: str) -> None:
        with self._lock:
            self._feedback = 0.0
            self._feedback_at = -np.inf

    def diagnostics(self) -> dict[str, JsonValue] | None:
        with self._lock:
            return dict(self._stats)

    # --- the bus thread ----------------------------------------------------------------------

    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        next_at = time.perf_counter()
        failures = 0
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self._step()
                failures = 0
            except (ConnectionError, OSError) as error:
                failures += 1
                with self._lock:
                    self._stats["retries"] += 1
                    if failures >= FAILURES_TO_FAULT:
                        self._fault = f"the GELLO on {self.device}: {error}"
                if failures >= FAILURES_TO_FAULT:
                    return
            elapsed = time.perf_counter() - started
            with self._lock:
                self._stats["cycles"] += 1
                self._stats["max_cycle_ms"] = max(self._stats["max_cycle_ms"], elapsed * 1e3)
            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            else:
                with self._lock:
                    self._stats["overruns"] += 1
                next_at = time.perf_counter()

    def _step(self) -> None:
        assert self._bus is not None and self._clock is not None
        present = self._bus.read()
        time_ns = self._clock.now_ns()
        raw = mapping.raw_angle(present.counts)
        joints = mapping.joint_angles(raw, self._offsets, self.signs)
        trigger_angle = mapping.trigger_angle(raw, self._offsets, self.gain)
        trigger_torque = feedback = 0.0
        if self.force_feedback:
            feedback = self._feedback_now(1.0 / self.rate_hz)
            opening = mapping.follower_opening(trigger_angle)
            low, high = mapping.LEADER_GRIPPER_RANGE
            inside = low < trigger_angle < high
            rate = OPENING_SLOPE * self.gain * present.velocity[-1] if inside else 0.0
            trigger_torque = mapping.trigger_spring(opening, rate) + feedback
            # toward released in the trigger's angle is the gain's sign on its motor
            motor = float(np.sign(self.gain)) * trigger_torque
            torques = np.r_[mapping.limit_torques(joints, self.signs), motor]
            self._bus.write_currents(mapping.currents(torques))
        cycle = _Cycle(time_ns, joints, mapping.trigger(trigger_angle), trigger_torque, feedback)
        with self._lock:
            self._cycle = cycle

    def _feedback_now(self, dt: float) -> float:
        """The client's torque, zero once stale, reached at no more than ``feedback_slew``."""
        with self._lock:
            fresh = time.monotonic() - self._feedback_at < self.feedback_timeout_s
            wanted = self._feedback if fresh else 0.0
        step = self.feedback_slew * dt
        self._applied_feedback += float(np.clip(wanted - self._applied_feedback, -step, step))
        return self._applied_feedback


def _ftdi_serial(device: str) -> str | None:
    """``FTBPFNFF`` from ``/dev/serial/by-id/usb-FTDI_..._FTBPFNFF-if00-port0``."""
    name = Path(device).name
    if "FTDI" not in name:
        return None
    return name.rsplit("_", 1)[-1].split("-")[0]
