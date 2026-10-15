# Archimedes analysis pack

Turns a unit's whole bench campaign into the figures and tables the IMSystems
test request asks for (`dyno/docs/1.2.2 Paladin Test Request.docx.pdf`).

```bash
./dyno/utilities/archimedes.sh gbx_1p2p0                 # the whole pack
./dyno/utilities/archimedes.sh gbx_1p2p0 --list          # what the logs hold
./dyno/utilities/archimedes.sh gbx_1p2p0 --only efficiency
./dyno/utilities/archimedes.sh gbx_1p2p0 --report        # + the LaTeX report
./dyno/utilities/archimedes.sh gbx_1p2p0 --report-only   # re-render it alone
```

Output lands in `<log_root>/analysis_pack/`: a PNG per figure, a CSV per table
(the customer asked for raw data), `report.txt` with every finding, and
`results.json` for anything downstream.

## The customer report

`--report` renders `<pack>/report/` — `report_generated.tex`, `sections/*.tex`
and `pics/` under stable semantic names — in the same shape and house style as
`dyno/src/analysis/tex`. Build it with `pdflatex report_generated.tex`.

It reads `results.json` and the CSVs the pack already wrote, never the HDF5, so
`--report-only` re-renders wording changes in a second without re-analysing.
Only figures the sections actually reference are copied.

**The report is deliberately flat.** It states what was measured, under what
conditions, and how the numbers were reduced — and stops. No compliance
verdicts, no pass/fail, no comparison against a rating, no attribution of
cause. "Slip occurred at 50.9 N·m measured at the output torque cell", not
"slip occurred at half the rated torque".

Measurement conditions that bear on how a number should be read are *not*
dropped — that would be worse than editorialising — but they are written as
conditions rather than conclusions. "The output shaft was held by the absorber
motor under position control rather than by a fixed mechanical ground" is a
condition; "the stiffness is therefore understated" is a conclusion, and it
belongs to whoever reads the report.

`dyno/sim/archimedes_analysis_test.py` enforces this: generated prose is
checked against a list of verdict words, and a back-driven result derived from
the forward sweep must be labelled as such in both its caption and its body
text.

The driver is regenerated on every render. To keep hand-written commentary,
copy it to a name of your own and edit that, or put the commentary in its own
file and `\input` it beside the section it belongs to.

## Why this is separate from `dyno/src/analysis`

The main framework analyses **one log directory** — `open_log` resolves to a
single `.hdf5` and a Processor sees one segment group. That is the right shape
for "analyse this run".

This is a different question: *a campaign*. The 1.2.0 efficiency sweep is four
files because a safety tripped three times, and the back-drive ramp is six
because the output kept walking into its position window. Those splits are
accidents of the bench day, and the next unit will break in different places.
So these analyzers take a whole multi-log `Dataset` instead, which is a
contract the `Processor` base class does not make — hence a sibling package
rather than six more processors.

The framework's `segment.Log`/`Segment` is reused for all HDF5 access, so
channel caching, `resolved_config`, `ports:` and tare handling are not
reimplemented.

## Running the next gearbox

Copy `units/TEMPLATE.yaml.example` to `units/<name>.yaml`, set `log_root`,
`torque_cap_nm` and `rated_torque_nm`, and run. Nothing in the Python needs
touching.

Specifically, you do **not** have to tell it:

- **how the tests were split across files.** Everything under `log_root` is
  found recursively and classified by segment name, not by folder name.
- **which runs are forward and which are back-drive.** Read off the data: the
  driving shaft is the one whose command sweeps (`dataset.direction_of`). A
  path that says `_bwd` while the input drove is reported, not obeyed.
- **that the back-drive pass arrived a week late.** Drop the logs anywhere
  under `log_root` and re-run.
- **that a resumed run repeated the level it died on.** The longer copy wins
  and the drop is listed in `report.txt`. Pass `--duplicates all` when the
  copies are deliberate repeats rather than restarts. Slip and breakaway ramps
  are exempt and every attempt is kept: each is an independent measurement of a
  stochastic release, and separate runs of a ramp plan share segment IDs
  because every launch restarts the numbering at ramp 1.

`exclude` is the one thing that does need thought: bring-up runs reuse the
plans' segment names, so a `calibration/` folder holding a `V0030` spin would
otherwise land in the customer's velocity ramp as a legitimate step.

## The three things to know before reading a number

1. **Frames.** `dut_*` is the INPUT (motor) shaft, `load_*` the OUTPUT.
   `dut_output_position` is in input rad despite the name; measured on the
   no-load sweep, the position spans divide at ~43.8.

2. **Every shuttle contains both directions of power flow.** The traverse is a
   bipolar sawtooth against a held torque: on one half-traverse the held torque
   opposes the motion (input drives output) and on the other it assists (output
   drives input). Averaging a segment whole mixes a forward efficiency with a
   back-driven one. Legs are split and classified by where the power actually
   flowed, which is why the forward sweep also yields a back-driven efficiency
   curve — reported separately, and not a substitute for the customer's
   back-drive plan, since the input is still the shaft holding the traverse.

3. **The output cell carries a zero offset.** On the 2026-09-17 campaign the
   direction-independent part of `load_torque` is flat against output angle at
   about +1.3 Nm. A real gravity term would go as sin(angle); flat means
   residual zero. It is 25% of the 5 Nm efficiency step, and uncorrected it
   pushes low-torque points over 100% efficient. `physics.cell_offsets`
   estimates it from the no-load traverses — the part that survives a direction
   flip is the zero, the part that reverses is Coulomb drag — and the figures
   say which correction was applied.

## Layout

| file | what it holds |
|---|---|
| `naming.py` | the segment-ID grammar (`V0300`, `E1500_P40_C3`, `K45`, `SLIP`, `BRK_N3`, …) |
| `dataset.py` | campaign-wide discovery, direction detection, de-duplication |
| `physics.py` | plateaus, traverse legs, shaft power, cell zero, robust pk-pk |
| `analyzers/velocity_ramp.py` | velocity ramp + no-load torque ripple + endstop travel |
| `analyzers/efficiency.py` | efficiency and loss curves, creep ratio, coverage |
| `analyzers/slip.py` | static slip torque and breakaway/stiction |
| `analyzers/stiffness.py` | wind-up hysteresis, K end-to-end and vs torque |
| `run.py` | the CLI, the unit files, and every file write |
| `report.py` | the customer-facing LaTeX report |

Adding an analyzer is one module under `analyzers/` with an
`analyze(ds, cfg) -> Result`, plus its name in `run.ORDER`.

## Back-drive, and the next two units

`dyno/docs/archimedes_backdrive_and_next_units.md` covers what the back-drive
half of the test list measured and what changes when 1.2.1 and 1.2.2 arrive.
The short version: 1.2.0's back-drive list is complete — velocity ramp,
efficiency, static slip and breakaway — and moving to a new unit is one YAML
file, but measure slip *first*, because the efficiency range and the stiffness
targets are both derived from it.

One thing that did *not* adapt on its own, and will not on the next unit
either: the back-drive ramp plans emit segment IDs the forward ones do not
(`SLIP_P1`, `BRK_N3`, one segment per attempt), every launch of a ramp plan
restarts its numbering so separate runs share IDs, and a breakaway has to be
read on whichever shaft the ramp pushed. All three are handled now — see that
document's §3.1 for why each one failed silently rather than loudly.

## Known limits of the current bench setup

These are properties of the rig, not of the code, and the analyzers emit them
as findings rather than hiding them:

- **Stiffness is a lower bound.** The absorber holds the output with a position
  loop, not a ground, so its servo stiffness is in series with the drive.
- **Wind-up is a small difference of two large angles** — about 1% of the
  output travel on K45 — so the stiffness number is sensitive to the ratio
  calibration.
- **Train drag is charged to the drive.** The ~0.6 Nm output-referred Coulomb
  drag of the whole train is inside every efficiency number; it is reported
  beside the curves so a reader can see how much of the low-torque loss is the
  bench.
