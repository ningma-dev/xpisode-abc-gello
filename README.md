# xpisode-abc-gello

abc's [GELLO](https://github.com/amazon-far/abc/tree/main/deploy/gello) leader arm, seven XL330s on
a U2D2, as an [xpisode](../xpisode) device. It reads like `xpisode_i2rt:YamLeader`: an `input`
stream with `joint_pos` (six joints, rad, in the YAM follower's convention) and `trigger`
(0 released, 1 squeezed), so a teleop client written for the YAM leader drives a follower from it.
The logic is abc's, ported (see `NOTICE`).

## Set up a GELLO

Once per GELLO, after it is built (and again after a horn is taken off and put back):

1. **IDs**, one motor at a time as it is wired (`python -m xpisode_abc_gello.ids`): each arm its
   own ten, M0 (base) .. M6 (trigger), at 2 000 000 baud, Return Delay Time 0. abc runs at
   4 000 000; there, a longer lead to the base motor lost about 1% of its replies, and none in
   10 000 at 2 000 000. The factory return delay, 500 us a reply, holds the bus to about 190 Hz.
2. **The port**: the U2D2's FTDI latency timer at 1 ms (a udev rule `ATTR{latency_timer}="1"`
   for `ftdi_sio`) and the user in `dialout`. The device refuses otherwise, as abc does.
3. **Calibration** (`python -m xpisode_abc_gello.calibrate`, motors unpowered): the zero pose,
   then each joint turned about 45 degrees the way the YAM's joint turns positive, then the
   trigger squeezed. It writes each joint's zero and direction and the trigger's ends to a file
   the device reads, and refuses a step where another motor moved instead (IDs on the wrong
   joints). abc instead assumes its own build's directions (its two profiles differ in joint 4)
   and must be started in the zero pose each time; a joint the wrong way round reads mirrored, and
   on the YAM's shoulder or elbow, which only turn one way from zero, it stays at its limit.
4. Check it (`python -m xpisode_abc_gello.check --calibration FILE`): bus rate, retries, the joints.

## Opening it

Any pose: the encoders are absolute within a turn, so each joint takes the turn that puts it
nearest the middle of its range (all are well under a turn wide). The file must be this
GELLO's: the device compares its U2D2 serial and servo IDs. The pose it opened in goes into
the Episode header with the calibration's path and date.

## In a station

```toml
[[devices]]
name = "leader_l"
exclusive = ["serial:FTBPFNFF"]
command = ["<venv>/bin/python", "-m", "xpisode_device.cli", "run", "--station", "{station}",
           "--name", "{name}", "--factory", "xpisode_abc_gello:GelloLeader", "--config",
           '{"device": "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBPFNFF-if00-port0", "servo_ids": [10, 11, 12, 13, 14, 15, 16], "calibration": "tuned/gello_leader_l.json"}']
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
