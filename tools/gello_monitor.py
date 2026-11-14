"""Live table of every GELLO servo. Torque stays off; turn joints by hand. Ctrl+C to quit.

  ~/xpisode/repos/abc/.venv/bin/python ~/gello_monitor.py          # arm 1L (IDs 10-16)
  ~/xpisode/repos/abc/.venv/bin/python ~/gello_monitor.py 1R       # IDs 20-26, etc.
  press Enter at any time to re-zero the "change" column
"""

import select
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORT, BAUD = "/dev/ttyUSB0", 2_000_000
ARM = (sys.argv[1] if len(sys.argv) > 1 else "1L").upper()
BASE = {"1L": 10, "1R": 20, "2L": 30, "2R": 40}[ARM]
IDS = [BASE + j for j in range(7)]
NAMES = ["base yaw", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3", "trigger"]
A_TORQUE, A_HWERR, A_POS, A_VOLT, A_TEMP = 64, 70, 132, 144, 146

ph, port = PacketHandler(2.0), PortHandler(PORT)
if not (port.openPort() and port.setBaudRate(BAUD)):
    sys.exit(f"Cannot open {PORT} at {BAUD}. Is another program (teleop, Wizard) using it?")


def s32(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def read(mid):
    pos, r, _e = ph.read4ByteTxRx(port, mid, A_POS)
    if r != COMM_SUCCESS:
        return None
    volt, _, _ = ph.read2ByteTxRx(port, mid, A_VOLT)
    temp, _, _ = ph.read1ByteTxRx(port, mid, A_TEMP)
    hw, _, _ = ph.read1ByteTxRx(port, mid, A_HWERR)
    return s32(pos), volt / 10, temp, hw


for mid in IDS:
    ph.write1ByteTxRx(port, mid, A_TORQUE, 0)
zero = {}
try:
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            zero = {}
        rows = []
        for j, mid in enumerate(IDS):
            d = read(mid)
            if d is None:
                rows.append(f" M{j}  {mid:>3}  {NAMES[j]:<9}  \033[31mNO RESPONSE\033[0m")
                continue
            pos, volt, temp, hw = d
            zero.setdefault(mid, pos)
            deg = pos * 360 / 4096
            chg = (pos - zero[mid]) * 360 / 4096
            bar = "#" * min(30, int(abs(chg) / 6))
            hl = "\033[1;32m" if abs(chg) > 10 else ""
            err = f"  \033[31mHW ERROR 0x{hw:02x}\033[0m" if hw else ""
            rows.append(f"{hl} M{j}  {mid:>3}  {NAMES[j]:<9}  {deg:8.1f}°  {chg:+7.1f}°  {volt:4.1f} V  {temp:3d} °C  "
                        f"{bar}\033[0m{err}")
        print("\033[H\033[2J", end="")
        print(f"GELLO {ARM}  {PORT} @ {BAUD}   torque OFF   Enter = re-zero   Ctrl+C = quit\n")
        print(" joint ID   name         angle    change    volt   temp")
        print("\n".join(rows))
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    port.closePort()
