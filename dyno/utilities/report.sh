#!/usr/bin/env bash
set -e
# Bootstrap and render a LaTeX report for a folder of experimental data.
#
#   report.sh                     # the current directory is the report root
#   report.sh path/to/report      # that directory is the report root
#   report.sh path/to/report -q   # extra flags pass through to the renderer
#
# Safe to run repeatedly: it creates what is missing and leaves everything else
# alone. The two bootstrap steps only ever fire on a fresh report root.
#
# Path-like arguments are resolved against the CALLER's directory before the cd
# below moves to the repo root, for the same reason analyze.sh does it: a
# relative path would otherwise be silently reinterpreted against the wrong
# root. Flags never name an existing path, so they pass through untouched.
args=()
root=''
for a in "$@"; do
  if [[ "$a" != -* && -e "$a" ]]; then
    a="$(realpath "$a")"
    [[ -z "$root" ]] && root="$a"
  fi
  args+=("$a")
done

# No positional argument means "here", which has to be captured before the cd.
if [[ -z "$root" ]]; then
  root="$PWD"
  args=("$root" "${args[@]}")
fi

if [[ ! -d "$root" ]]; then
  echo "report.sh: $root is not a directory" >&2
  exit 1
fi

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../.."

# -- 1. the manifest -------------------------------------------------------
# Everything hangs off this file: it marks the report root, and its paths are
# what let logs sit at any depth. A skeleton is enough to render, so a fresh
# root is never a hard error -- it is a file to fill in.
manifest="$root/report.yaml"
if [[ ! -f "$manifest" ]]; then
  cat > "$manifest" <<'YAML'
# Report manifest. This file's directory is the report root: sections/ and
# pics/ are written beside it, and every path below is relative to it, at any
# depth. Render with:  dyno/utilities/report.sh <this directory>

# What kind of device this report is about. Chooses the wording of the
# introduction, the reading order of the sections, and which rows the summary
# table carries. One of: motor (the default), gearbox.
# type: gearbox

# The logs this report covers. Available sections, with their roles:
#
#   torque_response: <log>          from the current_torque processor
#   inertia:                        from the inertia processor
#     full: <log>
#     structure: <log>              optional, the DUT-disconnected run
#   torque_ripple:                  from the cogging processor
#     drive_on: <log>
#     drive_off: <log>              optional
#   running_torque:                 from the running_torque processor
#     drive_on: <log>
#     drive_off: <log>              optional
#   backlash: <log>                 from the backlash processor
#   efficiency: <log>               from the efficiency processor
#
# List only the sections this report covers. Nothing is required and nothing
# has to be listed in order: a gearbox report that is backlash and efficiency
# alone renders as exactly those two sections, and the summary table carries
# exactly their rows.
#
# A section with no variants can be written as a bare path instead of a
# mapping, and drive_off may always be left out -- the tables and figures drop
# the on/off distinction rather than leaving a hole.
#
# Example (motor):
#
#   sections:
#     torque_response: blocked_rotor
#     inertia:
#       full: inertia/full_system
#       structure: inertia/only_coupler
#     torque_ripple:
#       drive_on: torque_ripple/basic
#     running_torque: running_torque/backdriven
#
# Example (gearbox):
#
#   sections:
#     backlash: backlash
#     efficiency: efficiency

sections: {}

# ---------------------------------------------------------------------------
# Everything below is optional. Uncomment what applies.
# ---------------------------------------------------------------------------

# Adds sentences to the generated introduction.
# title: My Motor
# author: Paladin Engineering
# dut:
#   supplier: Paladin Engineering
#   topology: transverse-flux
#   pole_pairs: 24
#   drive: Elmo Gold Drum 100A/100V
#   supply_V: 52.2
#   torque_cell_Nm: 20

# For type: gearbox, the DUT block takes a ratio and a topology instead, and
# the chain either side of it is described here. A gearbox sits between two
# machines rather than against one, and which end is which is what says how to
# read every torque in the report -- so it is stated rather than inferred.
# Both sides are optional; whatever is given is written into the introduction.
#
# dut:
#   supplier: Paladin Engineering
#   topology: cycloidal
#   gear_ratio: 26
# drivetrain:
#   input:                          # the high-speed side
#     motor: 1 kW absorber motor
#     torque_cell_Nm: 20
#   output:                         # the low-speed side
#     motor: 5 kW absorber motor
#     torque_cell_Nm: 100

# Which log feeds each cross-log row of the summary table. Report decisions,
# not data ones -- the drive-on and drive-off runs measure different things.
# Defaults to drive_on.
# summary:
#   ripple_from: drive_on
#   running_from: drive_on

# How a drive reports current, per device. Only needed for logs recorded
# before drive_params.current_units existed in the dyno config. This is a unit
# LABEL -- nothing anywhere converts between peak and rms.
# current_units:
#   DUT: peak
YAML
  echo "report.sh: wrote a skeleton $manifest"
  echo "           fill in its sections: block, then run this again."
fi

# -- 2. the logo -----------------------------------------------------------
# A missing \includegraphics is a fatal pdflatex error rather than a blank
# space, so the driver omits the byline logo when the file is absent. Copying
# it in up front means a fresh report gets the letterhead without the operator
# having to know that.
logo_src="$here/assets/paladinLogo.png"
logo_dst="$root/pics/paladinLogo.png"
if [[ ! -f "$logo_dst" ]]; then
  if [[ -f "$logo_src" ]]; then
    mkdir -p "$root/pics"
    cp "$logo_src" "$logo_dst"
    echo "report.sh: copied paladinLogo.png into $root/pics/"
  else
    echo "report.sh: no logo at $logo_src; the report will build without one" >&2
  fi
fi

# -- 3. render -------------------------------------------------------------
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  .venv/bin/python -m dyno.src.analysis.tex "${args[@]}"
