"""Scan and assign IDs for the ABC GELLO leader (XL330, protocol 2.0).

  python gello_ids.py scan            # list every motor at every common baud
  python gello_ids.py set 1L M3       # the ONE unconfigured motor on the bus -> arm 1L's M3, 2M baud

ID = arm * 10 + joint: 1L = 10..16, 1R = 20..26, 2L = 30..36, 2R = 40..46. Baud 2_000_000: abc uses
4M, but a longer motor lead lost ~1% of replies there and none at 2M.
"""
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORT = "/dev/ttyUSB0"
TARGET_BAUD = 2_000_000
BAUD_CODE = {9600: 0, 57600: 1, 115200: 2, 1_000_000: 3, 2_000_000: 4, 3_000_000: 5, 4_000_000: 6}
SCAN_BAUDS = (57600, 4_000_000, 1_000_000, 115200, 2_000_000, 3_000_000, 9600)
ARM_BASE = {"1L": 10, "1R": 20, "2L": 30, "2R": 40}
TARGET_ID = {f"{arm}-M{i}": base + i for arm, base in ARM_BASE.items() for i in range(7)}
ADDR_ID, ADDR_BAUD, ADDR_TORQUE = 7, 8, 64
ADDR_RETURN_DELAY = 9  # factory 250 (500 us per reply): 7 motors cost 3.5 ms a sync read

ph = PacketHandler(2.0)
port = PortHandler(PORT)
if not port.openPort():
    sys.exit(f"Cannot open {PORT} (permission? see dialout group)")


def scan(bauds=SCAN_BAUDS):
    found = []
    for baud in bauds:
        port.setBaudRate(baud)
        time.sleep(0.05)
        data, res = ph.broadcastPing(port)
        if res == COMM_SUCCESS:
            for mid, (model, _fw) in sorted(data.items()):
                found.append((baud, mid, model))
    return found


def check(res, err, what):
    if res != COMM_SUCCESS:
        sys.exit(f"{what} failed: {ph.getTxRxResult(res)}")
    if err:
        sys.exit(f"{what} error: {ph.getRxPacketError(err)}")


def name_of(mid):
    return next((k for k, v in TARGET_ID.items() if v == mid), "?")


cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
if cmd == "scan":
    found = scan()
    if not found:
        print("No motors found. Is the hub switched on and the U2D2 TTL lead in a hub TTL port?")
    for baud, mid, model in found:
        ok = "OK" if baud == TARGET_BAUD and mid in TARGET_ID.values() else "needs setup"
        print(f"baud {baud:>7}  id {mid:>3}  model {model}  joint {name_of(mid)}  {ok}")
elif cmd == "set":
    joint = f"{sys.argv[2].upper()}-{sys.argv[3].upper()}"
    if joint not in TARGET_ID:
        sys.exit(f"Unknown arm/joint {joint}; use e.g. 'set 1L M3'")
    new_id = TARGET_ID[joint]
    done = {mid for b, mid, _ in scan((TARGET_BAUD,)) if mid in TARGET_ID.values()}
    if new_id in done:
        sys.exit(f"ID {new_id} ({joint}) already exists on the bus at {TARGET_BAUD}. Nothing changed.")
    cands = [(b, m) for b, m, _ in scan() if not (b == TARGET_BAUD and m in TARGET_ID.values())]
    if len(cands) != 1:
        sys.exit(f"Need exactly one unconfigured motor on the bus, found {len(cands)}: {cands}. Nothing changed.")
    baud, old = cands[0]
    port.setBaudRate(baud)
    check(*ph.write1ByteTxRx(port, old, ADDR_TORQUE, 0), "torque off")
    check(*ph.write1ByteTxRx(port, old, ADDR_RETURN_DELAY, 0), "return delay 0")
    check(*ph.write1ByteTxRx(port, old, ADDR_ID, new_id), "write ID")
    time.sleep(0.1)
    if baud != TARGET_BAUD:
        res, err = ph.write1ByteTxRx(port, new_id, ADDR_BAUD, BAUD_CODE[TARGET_BAUD])
        if res != COMM_SUCCESS:  # the reply may arrive at the new baud and be lost
            print("(no reply after baud change; verifying...)")
    time.sleep(0.2)
    port.setBaudRate(TARGET_BAUD)
    time.sleep(0.05)
    model, res, err = ph.ping(port, new_id)
    check(res, err, f"verify ping ID {new_id} at {TARGET_BAUD}")
    print(f"{joint}: id {old}@{baud} -> id {new_id}@{TARGET_BAUD}  (model {model})  OK")
else:
    sys.exit(__doc__)
port.closePort()
