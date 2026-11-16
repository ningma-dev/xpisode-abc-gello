"""Give a new GELLO's motors their IDs, one motor at a time, as it is wired up.

    python -m xpisode_abc_gello.ids --device /dev/serial/by-id/...  --ids 10 11 12 13 14 15 16 scan
    python -m xpisode_abc_gello.ids --device ... --ids ... set M3

``--ids`` are the seven the GELLO will have, M0 (base) .. M6 (trigger). ``scan`` lists every
motor on the bus at every common baud rate. ``set Mk`` takes the one motor on the bus that is
not yet one of ``--ids`` at ``--baudrate`` (a new motor: ID 1 at 57600) and gives it the ID of
joint k, that baud rate and a zero return delay (the factory 500 us a reply holds the bus to
about 190 Hz). With more or fewer than one such motor it changes nothing. Wire one more motor,
set it, and so on; the base motor M0 is on its own branch of the hub.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

BAUD_CODE = {9600: 0, 57600: 1, 115200: 2, 1_000_000: 3, 2_000_000: 4, 3_000_000: 5, 4_000_000: 6}
SCAN_BAUDS = (57600, 4_000_000, 1_000_000, 115200, 2_000_000, 3_000_000, 9600)
ADDR_ID, ADDR_BAUD, ADDR_RETURN_DELAY, ADDR_TORQUE = 7, 8, 9, 64


class Motors:
    def __init__(self, device: str) -> None:
        self.packets = PacketHandler(2.0)
        self.port = PortHandler(device)
        if not self.port.openPort():
            sys.exit(f"cannot open {device} (is the user in the dialout group?)")

    def scan(self, bauds: Sequence[int] = SCAN_BAUDS) -> list[tuple[int, int, int]]:
        found = []
        for baud in bauds:
            self.port.setBaudRate(baud)
            time.sleep(0.05)
            data, result = self.packets.broadcastPing(self.port)
            if result == COMM_SUCCESS:
                found += [(baud, motor, model) for motor, (model, _) in sorted(data.items())]
        return found

    def write(self, motor: int, address: int, value: int, what: str) -> None:
        result, error = self.packets.write1ByteTxRx(self.port, motor, address, value)
        if result != COMM_SUCCESS or error:
            sys.exit(f"{what} failed: {self.packets.getTxRxResult(result)} (error {error}); "
                     "nothing more changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xpisode_abc_gello.ids", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ids", type=int, nargs=7, required=True)
    parser.add_argument("--baudrate", type=int, default=2_000_000, choices=sorted(BAUD_CODE))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan")
    set_ = commands.add_parser("set")
    set_.add_argument("joint", help="M0 .. M6")
    args = parser.parse_args(argv)
    ids, baud = list(args.ids), args.baudrate
    motors = Motors(args.device)
    try:
        if args.command == "scan":
            found = motors.scan()
            if not found:
                print("no motors: is the hub switched on, and the U2D2's TTL lead in a hub port?")
            for b, motor, model in found:
                done = b == baud and motor in ids
                where = f"M{ids.index(motor)}" if done else "not set up"
                print(f"baud {b:>7}  id {motor:>3}  model {model}  {where}")
            return 0
        joint = args.joint.upper()
        if joint not in {f"M{i}" for i in range(7)}:
            sys.exit(f"unknown joint {args.joint}; M0 .. M6")
        new = ids[int(joint[1])]
        if new in {m for _, m, _ in motors.scan((baud,))}:
            sys.exit(f"ID {new} ({joint}) is already on the bus at {baud}; nothing changed")
        new_motors = [(b, m) for b, m, _ in motors.scan() if not (b == baud and m in ids)]
        if len(new_motors) != 1:
            sys.exit(f"need exactly one motor not set up on the bus, found {new_motors}; "
                     "nothing changed")
        old_baud, old = new_motors[0]
        motors.port.setBaudRate(old_baud)
        motors.write(old, ADDR_TORQUE, 0, "torque off")
        motors.write(old, ADDR_RETURN_DELAY, 0, "return delay 0")
        motors.write(old, ADDR_ID, new, "ID")
        time.sleep(0.1)
        if old_baud != baud:  # the reply comes at the new rate and may be lost: verified below
            motors.packets.write1ByteTxRx(motors.port, new, ADDR_BAUD, BAUD_CODE[baud])
        time.sleep(0.2)
        motors.port.setBaudRate(baud)
        time.sleep(0.05)
        model, result, error = motors.packets.ping(motors.port, new)
        if result != COMM_SUCCESS or error:
            sys.exit(f"ID {new} does not answer at {baud} after the change")
        print(f"{joint}: id {old} at {old_baud} -> id {new} at {baud} (model {model})")
        return 0
    finally:
        motors.port.closePort()


if __name__ == "__main__":
    sys.exit(main())
