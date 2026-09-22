#!/usr/bin/env python3
"""SO-101 bus scan and servo ID reassignment.

Purpose:
1. Scan every servo that actually answers on a given serial port (ID, model
   number, firmware version) without requiring all 6 motors to be present
   (connect(handshake=False) skips the motor check). Useful when diagnosing
   "0 motors respond", "gripper read fails" or scrambled IDs.
2. Optionally reassign an ID (EEPROM address 5), for example changing a
   scrambled ID 7 back to 6.

Usage:

    # 1) Read-only scan (safe; do this first)
    python scripts/scan_so101_bus.py --port /dev/ttyACM0
    python scripts/scan_so101_bus.py --port /dev/ttyACM1

    # 2) Reassign an ID: current ID 7 -> new ID 6 (read the safety notes!)
    python scripts/scan_so101_bus.py --port /dev/ttyACM0 --set-id 7:6

    # 3) Several at once
    python scripts/scan_so101_bus.py --port /dev/ttyACM0 --set-id 7:6 --set-id 0:1

Safety notes (read before changing any ID):
- Strongly recommended: connect only a single motor to the bus while changing
  its ID. If several motors share the same current ID, the write hits all of
  them, and a new ID that collides with another motor corrupts the bus.
- Before writing, the script disables torque (Torque_Enable=0) and unlocks the
  EEPROM (Lock=0).
- Power-cycle the servo after each ID change before touching the next one.
- This is a permanent EEPROM write; a wrong value can make the robot unable to
  identify the servo, so proceed carefully.
- Afterwards, re-scan with ``lerobot-record`` or this script and confirm the
  six IDs are 1..6 with model=777.
"""
from __future__ import annotations

import argparse
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

JOINT_BY_ID = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}


def scan(bus: FeetechMotorsBus) -> dict[int, dict]:
    """Find responding IDs via broadcast_ping, then read model + firmware per ID."""
    found: dict[int, dict] = {}
    try:
        pinged = bus.broadcast_ping(num_retry=2) or {}
    except Exception as exc:  # protocol 1.0 has no broadcast_ping: ping one by one
        print(f"  broadcast_ping unavailable ({exc}); falling back to pinging 1..254 individually")
        pinged = {}
        for id_ in range(254):
            try:
                if bus.ping(id_) is not None:
                    pinged[id_] = 0
            except Exception:
                pass
    ids = sorted(pinged)
    if ids:
        models = bus._read_model_number(ids, raise_on_error=False)
        fw = bus._read_firmware_version(ids, raise_on_error=False)
        for id_ in ids:
            found[id_] = {
                "model": models.get(id_),
                "firmware": fw.get(id_),
            }
    return found


def print_scan(port: str, found: dict[int, dict]) -> None:
    print(f"\n-- Scan result {port}: {len(found)} motor(s) responded --")
    for id_ in sorted(found):
        info = found[id_]
        joint = JOINT_BY_ID.get(id_, "?")
        model = info["model"]
        fw = info["firmware"]
        model_str = f"{model} (0x{model:04X})" if model is not None else "read failed"
        mark = ""
        if id_ in JOINT_BY_ID:
            if model != 777:
                mark = "  <-- wrong model (expected 777=STS3215)"
        else:
            mark = "  <-- outside 1..6 (SO-101 IDs are expected to be 1..6)"
        print(f"  ID={id_:>3d}  {joint:<14s} model={model_str}  fw={fw}{mark}")
    missing = [i for i in range(1, 7) if i not in found]
    if missing:
        print(f"  Missing expected motor ID(s): {missing}")
    else:
        print("  All six expected motors (1..6) are online.")


def set_servo_id(bus: FeetechMotorsBus, current_id: int, new_id: int) -> None:
    """Write new_id to the motor currently at current_id (EEPROM address 5)."""
    if not (1 <= new_id <= 254):
        raise SystemExit(f"new ID must be within 1..254, got {new_id}")
    print(f"\nReassigning ID: {current_id} -> {new_id} (permanent EEPROM write)")
    print("    Steps: disable torque + unlock EEPROM (Lock=0) + write ID + re-scan")
    name = f"m{current_id}"
    bus.motors[name] = Motor(current_id, "sts3215", MotorNormMode.DEGREES)
    try:
        bus.disable_torque(current_id, num_retry=1)
    except Exception as exc:
        print(f"  WARN: disable torque failed (continuing): {exc}")
    try:
        bus.write("Lock", name, 0, num_retry=1)
    except Exception as exc:
        print(f"  WARN: Lock=0 failed (continuing): {exc}")
    try:
        bus.write("ID", name, new_id, num_retry=1)
        print("  Write succeeded. Power-cycle the servo, then re-scan to confirm.")
    except Exception as exc:
        raise SystemExit(f"  Writing the ID failed: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 bus scan / servo ID reassignment")
    ap.add_argument("--port", default="/dev/ttyACM0", help="serial port (default /dev/ttyACM0)")
    ap.add_argument("--set-id", action="append", default=[], metavar="CUR:NEW",
                    help="reassign a servo ID, repeatable, e.g. 7:6; connect only one motor first")
    args = ap.parse_args()

    bus = FeetechMotorsBus(port=args.port, motors=dict(MOTORS))
    try:
        print(f"Connecting to {args.port} (motor check skipped, scan only) ...")
        bus.connect(handshake=False)
        print("Port open.")

        if args.set_id:
            found = scan(bus)
            print_scan(args.port, found)
            for spec in args.set_id:
                cur, _, new = spec.partition(":")
                if not cur.isdigit() or not new.isdigit():
                    raise SystemExit(f"--set-id format must be CUR:NEW, got {spec!r}")
                set_servo_id(bus, int(cur), int(new))
                time.sleep(0.2)
                found = scan(bus)
                print_scan(args.port, found)
        else:
            found = scan(bus)
            print_scan(args.port, found)
            print("\nHint: once you know the IDs, use --set-id CUR:NEW to fix them (one motor at a time).")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass
    print("\nScan finished.")


if __name__ == "__main__":
    sys.exit(main())
