# xpisode-abc-gello

abc's [GELLO](https://github.com/amazon-far/abc/tree/main/deploy/gello) leader arm, seven XL330s on
a U2D2, as an [xpisode](../xpisode) device. It reads like `xpisode_i2rt:YamLeader`: an `input`
stream with `joint_pos` (six joints, rad, in the YAM follower's convention) and `trigger`
(0 released, 1 squeezed), so a teleop client written for the YAM leader drives a follower from it.
The logic is abc's, ported (see `NOTICE`).

## Set up a GELLO

1. Motor IDs: each arm its own ten, `arm * 10 + joint` (1L = 10–16, 1R = 20–26, 2L = 30–36,
   2R = 40–46), at 2 000 000 baud, and Return Delay Time (address 9) 0. Set them one motor
   at a time. The factory return delay, 500 µs a reply, holds the bus to about 190 Hz.
   abc runs at 4 000 000; there, a longer lead to the base motor lost about 1% of its replies,
   and none in 10 000 at 2 000 000. At 2M a sync read takes about 2 ms and the bus about
   360 Hz; with short, stock leads 4M gives about 1 ms and 400 Hz (`baudrate` in the config).
2. The U2D2's FTDI latency timer at 1 ms, and the port readable by the station's user:
   `dialout`, and a udev rule `ATTR{latency_timer}="1"` for `ftdi_sio`. The device refuses
   otherwise, as abc does.
3. Check it without a station; the motors stay unpowered unless `--force-feedback`:

```bash
python -m xpisode_abc_gello.check --device /dev/serial/by-id/usb-FTDI_..._FTBPFNFF-if00-port0 \
    --ids 10 11 12 13 14 15 16
```

## Opening it

The zero is found again at every open, as abc does: each joint's offset is the whole number of
45° steps that puts it nearest the calibration pose, **the YAM's zero with the trigger released**.
So the GELLO must be in that pose when the device opens; a joint more than
`calibration_tolerance_deg` (15) away refuses the open and says which. The offsets go into the
Episode header (`offset_steps`), with how far each joint was from the pose.

## In a station

```toml
[[devices]]
name = "leader_l"
exclusive = ["serial:FTBPFNFF"]
command = ["<venv>/bin/python", "-m", "xpisode_device.cli", "run", "--station", "{station}",
           "--name", "{name}", "--factory", "xpisode_abc_gello:GelloLeader", "--config",
           '{"device": "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBPFNFF-if00-port0", "servo_ids": [10, 11, 12, 13, 14, 15, 16]}']
```

Beside it, the leader's control group holds `leader_l/trigger_feedback` (not a joint target: the
GELLO cannot be driven to a pose, so there is no `mirror`), the teleop client is
`xpisode_abc_gello.teleop`, and every `[[inputs]]` on `leader_l/input` buttons goes: the GELLO
has none, so another device (the other YAM leader, a foot pedal) gives switch and record.

## The teleop client

`xpisode_abc_gello.teleop` is xpisode's reference teleop client changed for a leader that cannot
be driven (`examples/gello_left.toml`):

- **A takeover does not jump.** Given the follower while a policy had it elsewhere, it keeps,
  per joint, the gap between follower and leader, and closes it only as the leader moves
  (`converge_rate` of the motion past `still_rad`, never past zero): a leader held still leaves
  the follower still, and once closed the mapping is one to one again. abc
  (`deploy/robot/gym/dagger_rollout.py`) instead keeps the end effector's offset for good, through
  IK; `converge_rate = 0` is that offset in joint space.
- **The gripper** joins by abc's rule: relative to the leader at the takeover, until opened 0.1
  past that (it snaps open) or squeezed fully.
- **The grip is felt.** While it follows, the follower's `gripper_eff` past `deadband`, while the
  gripper is nearly still, goes to the GELLO's `trigger_feedback` as `gain` times that. The
  defaults come from 80 recorded episodes: open and still under 0.2 N·m, pressing on something a
  steady 1.03, moving up to 2.4 from acceleration alone.
- A leader sample older than `stale_s` is not followed.

What the operator can feel is only as fine as the follower's grip: xpisode-i2rt's force limiter
holds about 1 N·m as soon as the gripper presses on something, so the trigger tells contact
from none more than it tells how hard.

## What it does each cycle

One thread owns the bus at `rate_hz` (400): one sync read of current, velocity and position of
all seven motors, then, with `force_feedback` (on by default, as in abc), one sync write of their
currents: abc's push-back past the soft joint limits, abc's trigger spring toward released, and
the `trigger_feedback` torque a client last sent (0 to `max_feedback_torque`, 0.15 N·m; back to
zero `feedback_timeout_s` after the last one, or when halted, at no more than `feedback_slew`).
`read` returns the newest cycle, stamped when its reply arrived; `poll_hz` (200) is how often the
host records it. `diagnostics` counts cycles, overruns, retries and the slowest cycle.

## Tools

`tools/` holds the bench scripts used while building a GELLO, run with any Python that has
`dynamixel-sdk`: `gello_ids.py` sets one new motor's ID, baud rate and return delay (one
factory motor on the bus at a time), `gello_monitor.py` shows every motor's angle live (to check
which ID is which joint), `gello_jointcheck.py` records which ID moves as each joint is turned.
