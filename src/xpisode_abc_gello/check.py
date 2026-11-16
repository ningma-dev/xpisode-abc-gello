"""Open the GELLO without a station and report what the device would publish.

    python -m xpisode_abc_gello.check --device /dev/serial/by-id/...FTBPFNFF... \\
        --ids 10 11 12 13 14 15 16 --calibration tuned/gello_leader_l.json --seconds 10 \\
        [--force-feedback]

Without ``--force-feedback`` the motors stay unpowered. It prints the joints and the trigger
twice a second, then the cycle rate, overruns and the read age the host would see.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from xpisode_device import MonotonicClock

from xpisode_abc_gello.device import GelloLeader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ids", type=int, nargs=7, required=True)
    parser.add_argument("--calibration", required=True, help="the file calibrate wrote")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rate-hz", type=float, default=400.0)
    parser.add_argument("--baudrate", type=int, default=2_000_000)
    parser.add_argument("--force-feedback", action="store_true")
    args = parser.parse_args()
    clock = MonotonicClock()
    gello = GelloLeader(
        args.device,
        args.ids,
        args.calibration,
        baudrate=args.baudrate,
        rate_hz=args.rate_hz,
        force_feedback=args.force_feedback,
    )
    gello.open(clock)
    try:
        info = gello.describe()
        print("facts:", info.device)
        started, last_print, ages, readings = time.monotonic(), 0.0, [], 0
        while (now := time.monotonic()) - started < args.seconds:
            reading = gello.read("input")
            if reading is not None:
                readings += 1
                ages.append((clock.now_ns() - reading.time_ns) / 1e6)
                if now - last_print > 0.5:
                    last_print = now
                    joints = np.rad2deg(np.asarray(reading.values[0]))
                    trig, torque = float(reading.values[1][0]), float(reading.values[2][0])
                    print(f"joints deg {np.array2string(joints, precision=1, suppress_small=True)}"
                          f"  trigger {trig:.3f}  trigger_torque {torque:+.4f} N*m")
            time.sleep(1.0 / info.streams["input"].poll_hz)
        stats = gello.diagnostics() or {}
        elapsed = time.monotonic() - started
        print(f"bus: {stats.get('cycles', 0) / elapsed:.0f} Hz, overruns {stats.get('overruns')}, "
              f"retries {stats.get('retries')}, slowest cycle {stats.get('max_cycle_ms', 0):.2f} ms")
        if ages:
            print(f"host reads: {readings / elapsed:.0f} Hz new samples, sample age "
                  f"median {np.median(ages):.2f} ms, p99 {np.percentile(ages, 99):.2f} ms")
    finally:
        gello.close()


if __name__ == "__main__":
    main()
