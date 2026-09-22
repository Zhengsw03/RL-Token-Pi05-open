#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check whether the SO-101 follower servo chain is complete (pings each servo
individually instead of letting lerobot require all of them to be online).

Usage:
    conda activate <your-env>
    python so101_tools/check_follower_motors.py               # defaults to /dev/ttyACM0
    python so101_tools/check_follower_motors.py /dev/ttyACM1  # check the leader instead

Prints whether each ID responds plus voltage/temperature when readable, and
concludes where the chain breaks.
"""
import sys
import warnings
warnings.filterwarnings('ignore')

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
motors = {n: Motor(i + 1, 'sts3215', MotorNormMode.DEGREES) for i, n in enumerate(NAMES)}

bus = FeetechMotorsBus(port=PORT, motors=motors)
print(f"-- Checking {PORT} --")
try:
    bus.connect(handshake=False)
except Exception as e:
    print(f"[FAIL] Cannot open the serial port: {type(e).__name__}: {str(e)[:150]}")
    print("       Check that the device is plugged in, that you have permission")
    print("       (`ls -l /dev/ttyACM*`) and that no other process holds it.")
    raise SystemExit(1)

alive = []
for mid in range(1, 7):
    v = bus.ping(mid, num_retry=1, raise_on_error=False)
    name = NAMES[mid - 1] if mid <= len(NAMES) else '?'
    if v is not None:
        alive.append(mid)
        extra = ''
        try:
            volt = bus.read('Present_Voltage', mid, normalize=False)
            temp = bus.read('Present_Temperature', mid, normalize=False)
            extra = f"  voltage {volt/10:.1f}V  temp {temp}C"
        except Exception:
            pass
        print(f"  ID {mid} ({name:14s}): OK  model={v}{extra}")
    else:
        print(f"  ID {mid} ({name:14s}): no response")

print()
if len(alive) == 6:
    print("CONCLUSION: [OK] all 6 servos online; recording/teleoperation can start.")
    rc = 0
else:
    missing = [m for m in range(1, 7) if m not in alive]
    first_missing = missing[0]
    print(f"CONCLUSION: [FAIL] missing ID(s) {missing}; the first missing one is ID {first_missing}")
    if alive and max(alive) + 1 == first_missing:
        print(f"            -> The chain breaks on the cable between ID {max(alive)} "
              f"and ID {first_missing}.")
        print(f"               Check that the incoming cable of ID {first_missing} "
              f"({NAMES[first_missing-1]}) is fully seated.")
    else:
        print("            -> The break is not contiguous; a servo ID may have been "
              "changed, or a whole cable harness is faulty.")
    rc = 1
bus.disconnect()
raise SystemExit(rc)
