#!/usr/bin/env bash
#
# SO101 serial-port permission fix
#
# Symptoms: PermissionError: [Errno 13] Permission denied: '/dev/ttyACM1'
#           serial.serialutil.SerialException: [Errno 13] could not open port /dev/ttyACM1
#
# Cause:  /dev/ttyACM* defaults to root:dialout 0660
#         (/lib/udev/rules.d/50-udev-default.rules:28  KERNEL=="tty[A-Z]*[0-9]", GROUP="dialout")
#         while the current user is not in the dialout group (which is empty).
#
# Usage:  sudo bash fix_so101_permission.sh
#
# What it does:
#   1. Adds the user to the dialout group (permanent, but existing sessions must
#      log in again for it to take effect).
#   2. Installs a udev rule using TAG+="uaccess" so the logged-in user gets ACL
#      read/write access immediately -- no re-login needed, and it survives
#      reboots and re-plugs.
#   3. Fallback chmod on device nodes that already exist.
#
set -uo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "error: root required. Run:  sudo bash $0" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo "$USER")}"
VENDOR="1a86"   # QinHeng CH343 (SO101 servo bus adapter board)
PRODUCT="55d3"
RULE_FILE="/etc/udev/rules.d/99-so101-serial.rules"

echo "==> Target user: $TARGET_USER"
echo

# ---------- 1. dialout group ----------
if id -nG "$TARGET_USER" 2>/dev/null | tr ' ' '\n' | grep -qx dialout; then
    echo "[1/3] $TARGET_USER is already in the dialout group, skipping"
else
    if usermod -aG dialout "$TARGET_USER"; then
        echo "[1/3] Added $TARGET_USER to the dialout group"
        echo "      (permanent, but sessions already logged in must log out and back in)"
    else
        echo "[1/3] usermod failed; continuing with the udev approach" >&2
    fi
fi

# ---------- 2. udev rule (uaccess = effective without re-login) ----------
echo "[2/3] Writing $RULE_FILE"
cat > "$RULE_FILE" <<'EOF'
# SO101 / SO-ARM serial adapter board (QinHeng CH343, 1a86:55d3)
# TAG+="uaccess" grants the active seat user read/write access through the logind
# ACL, independent of the dialout group, so it takes effect without re-login.
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", TAG+="uaccess", GROUP="dialout", MODE="0660"
EOF

udevadm control --reload-rules
udevadm trigger --subsystem-match=tty
sleep 1
echo "      udev rules reloaded and triggered"

# ---------- 3. Fallback permissions ----------
echo "[3/3] Fallback permissions on existing device nodes"
found=0
for p in /dev/ttyACM*; do
    if [ -e "$p" ]; then
        chmod a+rw "$p" && echo "      chmod a+rw $p"
        found=1
    fi
done
if [ "$found" -eq 0 ]; then
    echo "      no /dev/ttyACM* node present (the arm may not be plugged in)"
fi

echo
echo "================ Result ================"
ls -l /dev/ttyACM* 2>/dev/null || echo "(no /dev/ttyACM* node)"
echo
echo "Groups of $TARGET_USER:"
id -nG "$TARGET_USER" | tr ' ' '\n' | sed 's/^/  /'
echo
echo "Verify access through the ACL:"
getfacl -p /dev/ttyACM0 2>/dev/null | grep -E "^user|^group|^other" || true
echo
echo "Next step:"
echo "  python -c \"import serial; serial.Serial('/dev/ttyACM1'); print('OK')\""
echo "  If that prints OK, the permissions are fixed."
