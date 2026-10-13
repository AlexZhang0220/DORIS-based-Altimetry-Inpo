# DORIS VTEC and Altimetry Interpolation

Derive vertical total electron content (VTEC) from dual-frequency DORIS
observations, then interpolate it onto satellite altimetry ionospheric
pierce points (IPPs). The repository retains the original two-stage DORIS
workflow and a separate cycle-level altimetry interpolation workflow.

```text
DORIS RINEX + SP3 + SINEX
          │
          ▼
Stage 1: geometry and raw STEC ──► daily .pkl
          │
          ▼  UTC GIM reference
Stage 2: arc detection and leveling ──► daily VTEC .pickle
          │
          ▼  altimetry IPPs + cycle tables + GIM + SINEX
INPO: DORIS-to-altimetry interpolation ──► cycleNNN.pickle
```

## Setup

Use a Python environment with these runtime packages:
```bash
python -m pip install numpy pandas scipy pyproj georinex xarray numba tqdm astropy
```

Install `pytest` to run the regression checks. Astropy provides the historical
TAI/UTC conversion; it is required by the corrected workflow.
This repository does not download input data. Prepare DORIS RINEX, SP3,
SINEX and supported IGS IONEX files. INPO additionally requires daily
altimetry products and cycle-definition CSV files.

## Configure and run

Run the modules from the repository root. They use parameters defined in
their source files; there is no command-line date or stage selector.

1. Edit [`doris_vtec/main.py`](doris_vtec/main.py): set the satellite,
   `start_date`, `proc_days`, `block_days`, and all input/output roots.
   Defaults select Jason-3 (`ja3`) and 11 days starting 2018-01-01.
2. Review the processing settings in
   [`doris_vtec/config.py`](doris_vtec/config.py), then run both DORIS stages:

   ```bash
   python -m doris_vtec.main
   ```

3. Edit [`doris_inpo/main.py`](doris_inpo/main.py): set the satellite,
   `cycle_list`, data roots, cycle-table paths and worker count. Defaults
   select Jason-3 cycle 70. Review
   [`doris_inpo/config.py`](doris_inpo/config.py), then run:

   ```bash
   python -m doris_inpo.main
   ```

Change the example absolute data paths for your installation. Use new output directories for
validation: executing a pipeline can replace an existing product at the same
path. The satellite list in the example driver is not a guarantee of full
support; the retained detector is specific to Jason-3 (`L39`) with supported
STAREC/ALCATEL antennas.

## Products and interfaces

| Product | Path below its configured root | Content |
| --- | --- | --- |
| Stage 1 | `YEAR/DOYddd.pkl` | Observations, corrected time, geometry and raw STEC |
| Stage 2 | `YEAR/DOYddd.pickle` | Leveled DORIS STEC and VTEC |
| INPO | `cycle_unit/cycleNNN.pickle` | Altimetry rows with interpolated DORIS and reference GIM VTEC |

Products are pandas DataFrames readable with `pandas.read_pickle`. The
Stage 2 public schema remains these ten columns:

```text
obs_epoch, ipp_lat, ipp_lon, stec, vtec, elevation,
map_value, station_code, prn, pass_id
```

Here `stec` is the leveled STEC, not the raw Stage 1 value. Product identity
is also stored in DataFrame `attrs`; preserve these attributes when passing
products between stages.

## Time conventions

| Data or operation | Convention |
| --- | --- |
| DORIS `obs_epoch` after Stage 1 | Receiver-clock-corrected TAI |
| Altimetry `obs_epoch` and cycle boundaries | TAI input contract |
| DORIS/altimetry temporal matching | TAI against TAI |
| GIM map epochs and daily file selection | UTC |
| SINEX validity/reference queries | UTC |

GIM and SINEX queries convert TAI observations to UTC using historical leap
seconds. They do not apply a fixed modern offset. Altimetry and cycle-table
loaders assume their upstream timestamps already satisfy the TAI contract;
they do not automatically verify or convert the source time scale.

## Scientific behavior and correctness fixes

The original defaults remain: Jason-3 leveling ratio `0.925`, 8-point
Lagrange orbit interpolation, 10° observation elevation mask, minimum 30
observations per arc, 9-second time-gap rule and 0° reference-elevation
threshold. IPP geometry and the nonzero-distance `1/d` interpolation weights
are unchanged. No new interpolation estimator or attitude model is included.

- RINEX uses `SYS / SCALE FACTOR`; the complete receiver-clock field is read
  and the clock correction is applied once.
- Satellite and ground PCO offsets use their respective nominal LOS
  projections. Frequency, wavelength, dual-frequency signs and units remain
  unchanged; separate STAREC/ALCATEL constants are retained.
- Missing phases are removed before time segmentation; both detector fits
  reject non-finite inputs and residuals.
- SP3 records are matched to the observation PRN. Conflicting duplicates
  fail, and queries outside orbit coverage return NaN. Station coordinates
  are matched by SINEX validity interval; overlapping solutions fail.
- IONEX `9999` is missing data. The retained reader supports 13 maps on the
  71 × 73 grid; unsupported layouts fail. GIM interpolation remains
  Earth-fixed bilinear in space and linear in time. Queries outside coverage
  or across gaps longer than the declared positive `INTERVAL` return NaN.
- Cross-file GIM grids and intervals must agree. Conflicting shared-midnight
  maps use the file belonging to that UTC day; other map conflicts fail.
- INPO reads a 300-second DORIS margin around each cycle for the existing
  pass window. Altimetry target rows, the ±5-minute pass window, the strict
  `<400 s` point window and spatial thresholds are retained. Exactly
  coincident IDW observations use their finite-value mean.

`read_ionex_file` still returns `(maps, latitude_grid, longitude_grid)`;
`maps` is a dict-compatible object carrying `interval_seconds`. Preserve
that metadata when merging maps. `get_gim_vtec` assumes TAI query timestamps
by default; explicitly pass `observation_time_system="UTC"` for UTC inputs.

## Relation to the manuscript

The current manuscript revision describes iterative fit-and-split detection
in §3.2 (PDF page 8). This repository retains the legacy fixed two-stage
implementation: first, a residual jump above 10 cycles enables splitting at
`|z-score| > 3`; each surviving segment is refitted once and split at residual
jumps above 3.5 cycles. It does not repeat until convergence. Both use the
`5L1 − L2` observable and a fit with constant, time and `1/sin(elevation)`
terms. The original gross L1-jump screen (`>5e5`) applies only when a group
falls within one calendar date.

The manuscript's §3.1 (PDF page 7) writes PCO with a shared `sin(elevation)`
factor; this release corrects the endpoint projections to LOS geometry.
Consequently, this release should not be described as reproducing the
manuscript's exact arc population or reported percentages. No detector
replacement to enforce manuscript equivalence is included.

## Historical products and validation

Rebuild Stage 1 before Stage 2: Stage 2 rejects intermediates without the
corrected-TAI and LOS-PCO identity markers. Normal INPO processing requires
marked corrected-TAI VTEC. Its direct DORIS reader can read historical files,
but preserves unknown time identity and rejects mixed conventions. Existing
archives are not automatically migrated; keep them separate from rebuilt
outputs and do not relabel old products as newly validated ones.

```bash
python -m pip install pytest
python -m pytest tests -q
```

The preparation snapshot passed 149 regression tests, including PCO,
readers, time identity, orbit/station safeguards, GIM boundaries and INPO
edge cases. This is a recorded validation result, not a guarantee for every
input dataset or future revision. Tests use isolated temporary products.

See the [DORIS processing guide](doris_vtec/README.md) and
[altimetry interpolation guide](doris_inpo/README.md) for module details.
