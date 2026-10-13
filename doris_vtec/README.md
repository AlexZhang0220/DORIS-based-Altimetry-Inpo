# DORIS observations to VTEC

This module converts DORIS RINEX observations to relative slant TEC (STEC),
then levels each accepted arc against an IGS Global Ionospheric Map (GIM)
and converts it to vertical TEC (VTEC). Both stages keep the existing
GitHub module paths and daily pickle products.

The complete Stage 2 detector currently accepts Jason-3 (`L39`) with
STAREC or ALCATEL ground antennas. Satellite names recognized by the RINEX
reader alone do not establish support for the complete processing chain.

## Run the pipeline

Use the Python environment described in the [repository README](../README.md).
From the repository root, edit dates, satellite and input/output paths in
[`main.py`](main.py), then run:

```sh
python -m doris_vtec.main
```

`main.py` is a configured script; it has no command-line options. It runs
Stage 1 followed by Stage 2 and writes the selected daily products. Use a
separate output directory when comparing runs, because existing products
at those paths are overwritten.

The stages can also be called separately:

```python
from pathlib import Path
import pandas as pd

from doris_vtec.config import DEFAULT_STAGE2_CONFIG
from doris_vtec.stage1_obs2stec.pipeline import process_block
from doris_vtec.stage2_stec2vtec.pipeline import process_day

date = pd.Timestamp("2018-01-01")
obs_root = Path("/data/DORISObsStorage/ja3")

process_block(
    date, date,
    sinex_file=Path("/data/sinex/dpod2020_current.snx"),
    sp3_root=Path("/data/sp3/ja3"),
    rinex_root=Path("/data/rinexobs/ja3"),
    output_root=obs_root,
    satellite="ja3",
)

process_day(
    date=date,
    obs_root=obs_root,
    gim_root=Path("/data/IGSGIM"),
    out_root=Path("/data/DORISVTECStorage/ja3"),
    satellite="ja3",
    config=DEFAULT_STAGE2_CONFIG.copy(),
)
```

`process_block` uses inclusive start/end dates. `iter_blocks(start, end,
block_days)` controls how auxiliary data are loaded; it does not change
the daily output layout. Stage 2 `process_day` writes its result to disk;
read the output pickle to inspect a successful result.

## Inputs and daily products

| Stage | Input or product | Path relative to the configured root |
|---|---|---|
| 1 | DORIS RINEX | `rinex_root/YYYY/ja3rxYYDDD.001` for `satellite="ja3"` |
| 1 | SP3 | `sp3_root/*.sp3.001`, with `.bYYDDD.eYYDDD.` coverage in the filename |
| 1 | SINEX | Explicit `sinex_file` path |
| 1 | Intermediate observations | `output_root/YYYY/DOYDDD.pkl` |
| 2 | Intermediate observations | `obs_root/YYYY/DOYDDD.pkl` |
| 2 | IGS IONEX | `gim_root/YYYY/IGS0OPSFIN_YYYYDDD*GIM.INX` or classic `igsgDDD0.YYi*` |
| 2 | Leveled VTEC | `out_root/YYYY/DOYDDD.pickle` |

Supply extracted text inputs. The IONEX reader expects the legacy daily
IGS layout: **13 TEC maps, 71 latitudes and 73 longitudes**. Other map
layouts are rejected explicitly.

Stage 1 contains the following columns:

| Columns | Meaning / units |
|---|---|
| `obs_epoch`, `receiver_clock_offset` | Corrected TAI label; original receiver offset in seconds |
| `station_id`, `station_code`, `station_freq_shift`, `prn`, `antenna_type` | Observation and instrument identifiers |
| `L1`, `L2` | Phase in cycles, restored using RINEX scale factors |
| `C1`, `C2` | Code range in metres, restored using RINEX scale factors |
| `sat_x`, `sat_y`, `sat_z`, `sta_x`, `sta_y`, `sta_z` | ECEF coordinates in metres |
| `elevation`, `ipp_lat`, `ipp_lon` | Angles in degrees |
| `map_value` | Dimensionless slant-to-vertical mapping factor |
| `dion`, `stec` | Relative ionospheric range combination in metres; relative STEC in TECU |

Stage 2 preserves the public ten-column schema:

```text
obs_epoch, ipp_lat, ipp_lon, stec, vtec, elevation, map_value,
station_code, prn, pass_id
```

Here `stec` is **leveled** STEC in TECU, whereas Stage 1 `stec` is relative
STEC. `vtec = stec / map_value`. Rows without a final VTEC are excluded.

## Time and product identity

- Stage 1 adds `receiver_clock_offset` to the RINEX epoch once and stores
  the resulting TAI label in `obs_epoch`. Subsequent stages preserve it.
- SP3 interpolation uses the same physical time basis; the supplied orbit
  epochs must be compatible with TAI. The reader does not convert arbitrary
  SP3 time systems automatically.
- SINEX validity/reference epochs and GIM map epochs remain UTC labels.
  TAI observation times are converted to UTC for these lookups and for
  GIM daily-file selection. Astropy supplies the historical leap-second
  conversion, rather than a fixed number of seconds.
- UTC leap-second labels cannot be represented by pandas timestamps; the
  lookup conversion assigns them to the final nanosecond of their UTC day.
  Stored observation epochs remain continuous TAI.

Stage 1 products carry these `DataFrame.attrs` markers:

```python
obs_epoch_time_scale = "TAI"
obs_epoch_receiver_clock_corrected = True
pco_projection_model = "nominal_los_v1"
```

Stage 2 requires all three markers and preserves them in its output.
**Rebuild Stage 1 and Stage 2 from source inputs after updating older
products.** An old pickle with matching column names is insufficient;
manually adding markers does not provide evidence that corrections were
applied.

## Geometry, PCO and leveling

- Orbit positions use the original Lagrange interpolation, with up to
  eight neighboring epochs by default. Matching PRNs, finite coordinates
  and unique epochs are checked. Conflicting SP3 records raise an error;
  observations outside orbit coverage receive NaN coordinates.
- Each matched SINEX validity interval uses one representative station
  position, propagated with the original linear velocity formula. A
  change of valid solution starts a new coordinate group. Unmatched
  observations retain NaN coordinates; overlapping solutions raise an error.
- IPP geometry and the modified single-layer mapping function are unchanged.
- PCO uses separate endpoint projections onto the station-to-satellite
  LOS. The ground factor is local Up dot LOS, equivalent to
  `sin(elevation)`; the nominal satellite factor is outward WGS84 normal
  dot LOS, equivalently minus the nadir-axis projection. The existing
  two-frequency signs are retained. STEC corrections use metres; the
  cycle-slip `5L1 - L2` background uses PCO divided by wavelength in cycles.
- GIM interpolation remains Earth-fixed bilinear spatial interpolation
  and linear temporal interpolation. Missing `9999` values become NaN.
  Queries outside map coverage or across a gap larger than the declared
  IONEX `INTERVAL` return NaN. Exact map epochs remain usable.
- The maximum-elevation observation is the reference for each arc. With
  `r = range_ratio_ja3`, the unchanged leveling formula is:

  ```text
  delta_stec = r * GIM_VTEC_ref * map_value_ref - stec_ref
  leveled_stec = stec + delta_stec
  vtec = leveled_stec / map_value
  ```

## Arc detection and defaults

Configuration lives in [`config.py`](config.py); the detector lives in
[`stage2_stec2vtec/pass_detection`](stage2_stec2vtec/pass_detection).

| Rule | Current value / behavior |
|---|---|
| Minimum accepted arc size | `min_obs_count = 30` |
| Elevation mask | `elevation > 10°` by default |
| Time segmentation | Split when the consecutive gap is greater than 9 seconds |
| Missing phase | Filter non-finite L1/L2 before time segmentation |
| Gross L1 jump | Reject if maximum absolute consecutive jump is `> 5e5` cycles and the segment lies on one date |
| Fit model | Constant, elapsed time and `1 / sin(elevation)` |
| First split | If any absolute residual difference is `> 10` cycles, split at absolute z-score `> 3` |
| Refinement | Refit each retained segment once, then split at absolute residual difference `> 3.5` cycles |
| Reference elevation | `min_ref_elev = 0°`; reference must meet `>=` this value |
| Jason-3 GIM leveling ratio | `range_ratio_ja3 = 0.925` |

Both least-squares stages reject non-finite inputs or residuals. These
checks repair invalid-input handling without changing the fit or thresholds.

### Relation to the manuscript

The supplied *Assessment_DORIS_VTEC_Altimetry* revision, Section 3.2,
describes iterative cycle-slip refinement. This implementation uses the
fixed two-stage procedure above, including the 10-cycle/z-score gate,
one refit and the same-date gross-jump rule. It is therefore not an exact
implementation of the manuscript's iterative description.

The revision's Equation 4 also applies a shared elevation factor to PCO;
this implementation uses the separate endpoint LOS projections described
above. These differences should be reconciled before claiming that this
version reproduces the manuscript's results. The correctness update does
not replace the detector or alter its thresholds.
