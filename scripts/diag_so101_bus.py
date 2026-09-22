#!/usr/bin/env python
"""SO-101 follower (so101_follower) bus diagnostics.

Purpose: investigate failures such as

    ConnectionError: Failed to write 'Lock' on id_=5 ... [TxRxResult] Incorrect status packet!

raised by ``lerobot-record`` when connecting to the follower arm, and decide
whether motor 5 (wrist_roll) has a wiring/power problem or the bus simply has
intermittent communication errors.

Usage (run in a local terminal with the arm powered on and USB connected):

    # 1) Read-only stress test: 50 rounds of per-motor position/voltage/
    #    temperature reads, counting failures.
    python scripts/diag_so101_bus.py --port /dev/ttyACM0 --rounds 50

    # 2) Reproduce the failing write: replay the enable_torque sequence
    #    (Torque_Enable=1 followed by Lock=1) on wrist_roll (id 5). This
    #    briefly holds torque on joint 5, so make sure the arm is in a safe
    #    position first.
    python scripts/diag_so101_bus.py --port /dev/ttyACM0 --torque-test

Notes:
- Confirm /dev/ttyACM0 really is the follower before running (ports can swap
  after a reboot).
- Do not keep lerobot-record / lerobot-control running at the same time.
"""
from __future__ import annotations

import argparse
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, TorqueMode

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),  # the motor that fails
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def read_stress_test(bus: FeetechMotorsBus, rounds: int) -> None:
    """Repeatedly read Present_Position / Present_Voltage / Present_Temperature per motor."""
    failures = {name: 0 for name in MOTORS}
    voltage: dict[str, int | None] = {name: None for name in MOTORS}
    temperature: dict[str, int | None] = {name: None for name in MOTORS}
    sync_failures = 0

    print(f"\n-- Read stress test: {rounds} rounds, per-motor reads --")
    for i in range(rounds):
        # Same bulk sync read as get_observation (reported, not attributed to a motor)
        try:
            bus.sync_read("Present_Position")
        except Exception as e:  # noqa: BLE001
            sync_failures += 1
            if sync_failures <= 3:
                print(f"  [round {i}] sync_read failed as a whole: {e}")

        for name in MOTORS:
            try:
                bus.read("Present_Position", name)
            except Exception as e:  # noqa: BLE001
                failures[name] += 1
                if failures[name] <= 3:
                    print(f"  [round {i}] {name} read failed: {e}")
            if i == rounds - 1:  # collect voltage/temperature on the final round
                try:
                    voltage[name] = bus.read("Present_Voltage", name)
                except Exception:  # noqa: BLE001
                    voltage[name] = None
                try:
                    temperature[name] = bus.read("Present_Temperature", name)
                except Exception:  # noqa: BLE001
                    temperature[name] = None
        time.sleep(0.05)

    print(f"\n-- Read test result ({rounds} rounds per motor) --")
    for name, cnt in failures.items():
        mark = "  <-- FAILING" if cnt > 0 else ""
        print(f"  {name:14s} id={MOTORS[name].id}  failed {cnt:3d}/{rounds}{mark}")
    print(f"  sync_read failed as a whole {sync_failures}/{rounds} time(s)")
    print("\n-- Final round voltage / temperature (raw voltage: STS3215 unit 0.1V, so 84 ~ 8.4V) --")
    for name in MOTORS:
        v = voltage[name]
        t = temperature[name]
        v_str = f"{v / 10:.1f} V" if v is not None else "read failed"
        t_str = f"{t} C" if t is not None else "read failed"
        print(f"  {name:14s} id={MOTORS[name].id}  {v_str:>10s}   {t_str}")

    bad = [name for name, cnt in failures.items() if cnt > 0]
    if bad:
        print(f"\nCONCLUSION: {', '.join(bad)} failed reads. Check that motor, its wiring/connector and power first.")
    elif sync_failures:
        print("\nCONCLUSION: every per-motor read succeeded, but bulk sync_read still failed; the bus has intermittent errors.")
    else:
        print("\nCONCLUSION: all motor reads succeeded. The problem is likely write-only (intermittent); see the --torque-test result.")


def torque_test(bus: FeetechMotorsBus) -> None:
    """Replay the enable_torque write sequence on wrist_roll (id 5) to reproduce the failure."""
    name = "wrist_roll"
    print("\n-- Reproducing the enable_torque write sequence on wrist_roll (id=5) --")
    print("WARNING: writing Torque_Enable=1 briefly holds torque on joint 5; make sure the arm is in a safe position.")
    for attempt in range(5):
        # Exactly the same order as enable_torque
        try:
            bus.write("Torque_Enable", name, TorqueMode.ENABLED.value)
            bus.write("Lock", name, 1)
            print(f"  attempt {attempt}: Torque_Enable=1, Lock=1 -> OK")
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}: write failed -> {type(e).__name__}: {e}")
        finally:
            try:
                bus.write("Torque_Enable", name, TorqueMode.DISABLED.value)
                bus.write("Lock", name, 0)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.5)

    # Control group: run one identical write round on each remaining motor
    print("\n-- Control group: one enable_torque write round per remaining motor --")
    for mname in MOTORS:
        if mname == name:
            continue
        try:
            bus.write("Torque_Enable", mname, TorqueMode.ENABLED.value)
            bus.write("Lock", mname, 1)
            print(f"  {mname:14s} id={MOTORS[mname].id}: OK")
        except Exception as e:  # noqa: BLE001
            print(f"  {mname:14s} id={MOTORS[mname].id}: failed -> {type(e).__name__}: {e}")
        finally:
            try:
                bus.write("Torque_Enable", mname, TorqueMode.DISABLED.value)
                bus.write("Lock", mname, 0)
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 follower bus diagnostics")
    ap.add_argument("--port", default="/dev/ttyACM0", help="follower serial port (default /dev/ttyACM0)")
    ap.add_argument("--rounds", type=int, default=50, help="read-test rounds (default 50)")
    ap.add_argument("--torque-test", action="store_true", help="replay the wrist_roll enable_torque write sequence")
    args = ap.parse_args()

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    try:
        print(f"Connecting to {args.port} ...")
        bus.connect()  # pings all 6 motors and verifies firmware versions
        print("Connected (all 6 motors answered ping and firmware reads)")

        read_stress_test(bus, args.rounds)

        if args.torque_test:
            torque_test(bus)
        else:
            print("\nHint: add --torque-test to reproduce the failing write sequence.")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass
    print("\nDiagnostics finished.")


if __name__ == "__main__":
    sys.exit(main())
