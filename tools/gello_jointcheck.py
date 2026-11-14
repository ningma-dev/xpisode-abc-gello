"""Report which motor ID moves, in the order the user turns joints by hand. Torque stays off."""
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORT, BAUD = "/dev/ttyUSB0", 4_000_000
IDS = list(range(10, 17))
ADDR_POS, ADDR_TORQUE = 132, 64
THRESH_DEG = 20.0
TIMEOUT_S = float(sys.argv[1]) if len(sys.argv) > 1 else 180

ph, port = PacketHandler(2.0), PortHandler(PORT)
assert port.openPort() and port.setBaudRate(BAUD), "cannot open port"


def read_all():
    out = {}
    for mid in IDS:
        val, res, err = ph.read4ByteTxRx(port, mid, ADDR_POS)
        if res == COMM_SUCCESS and not err:
            out[mid] = val - (1 << 32) if val >= (1 << 31) else val
    return out


for mid in IDS:
    ph.write1ByteTxRx(port, mid, ADDR_TORQUE, 0)
base = read_all()
missing = [m for m in IDS if m not in base]
print(f"online: {sorted(base)}  missing: {missing}", flush=True)
order, t0 = [], time.time()
while time.time() - t0 < TIMEOUT_S and len(order) < len(IDS):
    now = read_all()
    deltas = {m: abs(now[m] - base[m]) * 360 / 4096 for m in now if m in base}
    if deltas:
        mid, d = max(deltas.items(), key=lambda kv: kv[1])
        if d > THRESH_DEG:
            others = {m: round(v) for m, v in deltas.items() if m != mid and v > 5}
            print(f"t={time.time() - t0:5.1f}s  ID {mid} (M{mid - 10}) moved {d:.0f} deg"
                  + (f"  [also moved: {others}]" if others else ""), flush=True)
            if mid not in order:
                order.append(mid)
            time.sleep(1.5)  # let the user finish this joint
            base = read_all()
    time.sleep(0.05)
print("order detected:", [f"ID{m}=M{m - 10}" for m in order], flush=True)
port.closePort()
