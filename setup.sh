#!/usr/bin/env bash
#
# Provision (or update) a machine to run the dyno stack.
#
# Safe to re-run: every step is idempotent, so this doubles as the update path
# for a client rig that already has an older install. Re-running is in fact
# REQUIRED after the venv is ever rebuilt, because file capabilities do not
# survive it (see the setcap section).
#
# Usage:
#   ./setup.sh                      # interface read from master_config.yaml,
#                                   #   or auto-detected if only one candidate
#   ./setup.sh --interface enp2s0   # name the EtherCAT NIC explicitly
#   ./setup.sh --verify             # check an existing install, change nothing
#
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$PWD"
CONFIG="dyno/config/master_config.yaml"
NM_CONF="/etc/NetworkManager/conf.d/99-ethercat-unmanaged.conf"
UNIT_SRC="deployment/ethercat-link@.service"
UNIT_DST="/etc/systemd/system/ethercat-link@.service"

IFACE=""
VERIFY_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interface|-i) IFACE="${2:?--interface needs a NIC name}"; shift 2 ;;
        --verify)       VERIFY_ONLY=1; shift ;;
        -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# 0. Resolve the EtherCAT NIC
#
# This CANNOT be hardcoded. "enp86s0" is a predictable name derived from PCI
# topology (bus 86, slot 0); a client machine with different hardware gets
# eno1, enp2s0, ... The name has to agree in three places - master_config.yaml,
# the NetworkManager unmanaged rule, and the systemd unit instance - so it is
# resolved once here and propagated to all three.
# --------------------------------------------------------------------------
config_iface() {
    [[ -f "$CONFIG" ]] || return 0
    awk '/^[[:space:]]*interface:/ {print $2; exit}' "$CONFIG"
}

# Wired NICs, excluding loopback, wireless, virtual devices, and whichever
# interface currently carries the default route (that is the machine's
# management/internet link, never the fieldbus).
detect_candidates() {
    local ifc primary
    primary="$(ip route show default 2>/dev/null | awk '/^default/ {print $5; exit}')"
    for ifc in /sys/class/net/*; do
        ifc="$(basename "$ifc")"
        [[ "$ifc" == "lo" ]] && continue
        [[ -d "/sys/class/net/$ifc/wireless" ]] && continue
        [[ -e "/sys/class/net/$ifc/device" ]] || continue   # skips veth/docker/bridges
        [[ "$ifc" == "$primary" ]] && continue
        echo "$ifc"
    done
}

if [[ -z "$IFACE" ]]; then
    IFACE="$(config_iface || true)"
    if [[ -n "$IFACE" && ! -e "/sys/class/net/$IFACE" ]]; then
        warn "$CONFIG names '$IFACE', which does not exist on this machine."
        IFACE=""
    fi
fi

if [[ -z "$IFACE" ]]; then
    mapfile -t CANDIDATES < <(detect_candidates)
    case "${#CANDIDATES[@]}" in
        0) die "No candidate EtherCAT NIC found. Pass one with --interface <name>.
       Wired interfaces seen: $(ls /sys/class/net | tr '\n' ' ')" ;;
        1) IFACE="${CANDIDATES[0]}"
           say "Auto-detected EtherCAT NIC: $IFACE" ;;
        *) die "Multiple candidate NICs: ${CANDIDATES[*]}
       Pick the one wired to the EtherCAT chain and re-run:
         ./setup.sh --interface <name>" ;;
    esac
fi

[[ -e "/sys/class/net/$IFACE" ]] || die "Interface '$IFACE' does not exist on this machine."
say "EtherCAT interface: $IFACE"

# --------------------------------------------------------------------------
# Verification, used both by --verify and as the final report of a full run.
# --------------------------------------------------------------------------
verify() {
    local rc=0
    say "Verifying installation"

    if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import pysoem' 2>/dev/null; then
        echo "  [ ok ] venv present, pysoem importable"
    else
        echo "  [FAIL] .venv missing or pysoem not importable"; rc=1
    fi

    local p missing=0
    for p in .venv/bin/python .venv/bin/python3 .venv/bin/python3.13; do
        [[ -e "$p" ]] || continue
        if ! getcap "$p" 2>/dev/null | grep -q cap_net_raw; then
            echo "  [FAIL] $p is missing capabilities (re-run ./setup.sh)"; missing=1
        fi
    done
    [[ $missing -eq 0 ]] && echo "  [ ok ] venv interpreters hold cap_net_raw/net_admin/sys_nice" || rc=1

    if [[ -f "$NM_CONF" ]] && grep -q "interface-name:$IFACE" "$NM_CONF"; then
        echo "  [ ok ] NetworkManager told to ignore $IFACE"
    else
        echo "  [FAIL] $NM_CONF missing or does not name $IFACE"; rc=1
    fi

    if systemctl is-enabled --quiet "ethercat-link@$IFACE.service" 2>/dev/null; then
        echo "  [ ok ] ethercat-link@$IFACE.service enabled (survives reboot)"
    else
        echo "  [FAIL] ethercat-link@$IFACE.service not enabled - link will be DOWN after reboot"; rc=1
    fi

    local state
    state="$(cat "/sys/class/net/$IFACE/operstate" 2>/dev/null || echo unknown)"
    if [[ "$state" == "up" ]]; then
        echo "  [ ok ] $IFACE is up"
    else
        echo "  [warn] $IFACE operstate=$state (chain unpowered or cable out?)"
    fi

    if ip -4 addr show "$IFACE" 2>/dev/null | grep -q 'inet '; then
        echo "  [warn] $IFACE has an IPv4 address - it should have none; IP traffic"
        echo "         does not belong on the fieldbus segment"
    fi

    if [[ $rc -eq 0 ]]; then
        say "All checks passed. Next: ./dyno/bus_scan.sh"
    else
        say "Some checks failed - see above."
    fi
    return $rc
}

if [[ $VERIFY_ONLY -eq 1 ]]; then
    verify
    exit $?
fi

# --------------------------------------------------------------------------
# 1. System packages
# --------------------------------------------------------------------------
command -v apt-get >/dev/null || die "This installer targets Debian/Ubuntu (apt not found)."

if ! command -v python3.13 >/dev/null; then
    say "Installing Python 3.13 from the deadsnakes PPA"
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.13 python3.13-venv
else
    echo "Python 3.13 already present: $(command -v python3.13)"
fi

# libxcb-cursor0 - PySide6/Qt xcb platform plugin
# python3.13-tk   - tkinter, used by test_manager.py and post_processor.py
# libcap2-bin     - setcap/getcap
say "Installing system dependencies"
NEEDED=()
for pkg in python3.13-venv python3.13-tk libxcb-cursor0 libcap2-bin; do
    dpkg -s "$pkg" >/dev/null 2>&1 || NEEDED+=("$pkg")
done
if [[ ${#NEEDED[@]} -gt 0 ]]; then
    sudo apt-get update
    sudo apt-get install -y "${NEEDED[@]}"
else
    echo "All system packages already installed."
fi

# --------------------------------------------------------------------------
# 2. Virtual environment
#
# --copies: install real python binaries (not symlinks) so the setcap below
# elevates only this venv's interpreters, never the system-wide one.
# --------------------------------------------------------------------------
say "Creating/updating the virtual environment"
mkdir -p dyno/logs          # dyno_paths.dyno_logs_directory (was: repo-root logs/)

if [[ ! -x .venv/bin/python ]]; then
    /usr/bin/python3.13 -m venv --copies .venv
else
    echo "Reusing existing .venv"
fi

# Call the venv's pip directly rather than sourcing activate: with `set -u` an
# activate script can trip on unset vars, and if the venv were ever missing,
# a bare `pip` would silently install into the SYSTEM python instead.
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# --------------------------------------------------------------------------
# 3. Capabilities
#
#   cap_net_raw   - pysoem's raw AF_PACKET EtherCAT socket
#   cap_net_admin - SOEM's ecx_setupnic() forces the NIC promiscuous via
#                   ioctl(SIOCSIFFLAGS), which the kernel gates behind
#                   CAP_NET_ADMIN. Without it pysoem raises the misleading
#                   "could not open interface <if>" even though the raw
#                   socket and bind both succeed.
#   cap_sys_nice  - SCHED_FIFO real-time scheduling for the cyclic loop
#
# --copies gives three *independent* binaries (python, python3, python3.13 -
# separate inodes, not symlinks). File capabilities attach to the inode, so all
# three must be capped: the launch scripts invoke .venv/bin/python.
#
# Capabilities do NOT survive a venv rebuild, which is why this runs on every
# invocation rather than only on first install.
# --------------------------------------------------------------------------
say "Granting EtherCAT capabilities to the venv interpreters"
for p in .venv/bin/python .venv/bin/python3 .venv/bin/python3.13; do
    [[ -e "$p" ]] || continue
    sudo setcap cap_net_raw,cap_net_admin,cap_sys_nice+ep "$p"
    echo "  $p -> $(getcap "$p" | cut -d' ' -f2-)"
done

# --------------------------------------------------------------------------
# 4. Fieldbus NIC: keep NetworkManager off it
#
# The EtherCAT master speaks raw EtherType 0x88A4 frames. If NetworkManager
# owns this NIC it will emit DHCP, IPv6 RA and ARP traffic onto the fieldbus
# segment, which at best adds jitter and at worst confuses slaves.
# --------------------------------------------------------------------------
say "Configuring NetworkManager to ignore $IFACE"
NM_WANT="[keyfile]
unmanaged-devices=interface-name:$IFACE"

if [[ -f "$NM_CONF" ]] && [[ "$(cat "$NM_CONF")" == "$NM_WANT" ]]; then
    echo "Already configured: $NM_CONF"
else
    printf '%s\n' "$NM_WANT" | sudo tee "$NM_CONF" >/dev/null
    echo "Wrote $NM_CONF"
    if systemctl is-active --quiet NetworkManager; then
        sudo nmcli general reload || sudo systemctl reload NetworkManager
    fi
fi

# --------------------------------------------------------------------------
# 5. Fieldbus NIC: bring the link up at boot
#
# Taking the NIC away from NetworkManager above has a side effect: nothing
# brings the link up any more, and SOEM cannot open a down interface. This
# templated unit does the one remaining thing - `ip link set up`, no
# addressing. Without it the rig works until the first reboot and then fails
# with "found no slaves", which is a genuinely confusing symptom.
# --------------------------------------------------------------------------
say "Installing the link-up service for $IFACE"
[[ -f "$UNIT_SRC" ]] || die "$UNIT_SRC missing from the repo."

if [[ -f "$UNIT_DST" ]] && cmp -s "$UNIT_SRC" "$UNIT_DST"; then
    echo "Unit already installed and current."
else
    sudo cp "$UNIT_SRC" "$UNIT_DST"
    sudo systemctl daemon-reload
    echo "Installed $UNIT_DST"
fi

# `enable` is what makes this survive a reboot; --now also starts it here.
sudo systemctl enable --now "ethercat-link@$IFACE.service"

# --------------------------------------------------------------------------
# 6. Point the app at the resolved interface
# --------------------------------------------------------------------------
CURRENT="$(config_iface || true)"
if [[ -n "$CURRENT" && "$CURRENT" != "$IFACE" ]]; then
    say "Updating $CONFIG: $CURRENT -> $IFACE"
    sed -i "s/^\([[:space:]]*interface:[[:space:]]*\)$CURRENT\b/\1$IFACE/" "$CONFIG"
fi

verify
