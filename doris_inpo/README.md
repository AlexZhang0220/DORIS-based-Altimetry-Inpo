# DORIS VTEC at altimetry IPPs

This module interpolates leveled DORIS VTEC onto altimetry ionospheric
pierce points (IPPs). It loads one altimetry cycle, computes a separate
GIM comparison value, processes passes in parallel and saves a cycle
table. The interpolation remains adaptive-radius inverse-distance
weighting (IDW).

## Run a configured cycle

Use the environment described in the [repository README](../README.md)
and run from the repository root:

```sh
python -m doris_inpo.main
```

Before running, edit the configuration block in [`main.py`](main.py):

- `satellite` and `cycle_list`;
- `sinex_path`, `gim_root`, `doris_vtec_root`, `alt_vtec_root`;
- `dbav_root` and the two cycle-table CSV paths;
- `max_workers` if needed for the available machine.

This is a configured script, with no command-line options. The current
example uses Jason-3 cycle tables: `jason_3_f.csv` for cycles up to 227
and `jason_3_em_f.csv` thereafter. CSV columns consumed by the reader are
`cycle_number`, `start_day`, `start_time`, `end_date` and `end_time`.
Supply cycle times on the same TAI basis as the input observations.

Products at the selected output paths are overwritten. Use a separate
`dbav_root` for comparisons.

## Input and output contracts

| Input / product | Configured path pattern | Required data |
|---|---|---|
| Altimetry | `alt_vtec_root/YYYY/DOYDDD.pkl` | `obs_epoch`, `ipp_lat`, `ipp_lon`, `VTEC`, `pass` |
| DORIS VTEC | `doris_vtec_root/YYYY/DOYDDD.pickle` | Stage 2 product from [`doris_vtec`](../doris_vtec/README.md) |
| IGS GIM | `gim_root/YYYY/…` | Supported 13-map, 71×73 IONEX layout |
| Station coordinates | Explicit `sinex_path` | SINEX position, velocity and validity intervals |
| Cycle output | `dbav_root/cycle_unit/cycleNNN.pickle` | Input altimetry columns plus interpolation fields |

The altimetry reader removes missing `VTEC`, renames `pass` to
`pass_in_cycle` and drops `VTEC_ns` and `ascend` if present. Altimetry
`obs_epoch` is assumed to be TAI; it is not converted or clock-corrected
again by this module.

The saved table retains the altimetry fields and adds:

| Column | Meaning / units |
|---|---|
| `gim_vtec` | Interpolated GIM VTEC, TECU; no `0.925` scaling is applied here |
| `doris_vtec_idw` | DORIS VTEC interpolated to the altimetry IPP, TECU |
| `doris_points_count` | Number of spatial candidates selected |
| `doris_points_std` | Standard deviation of selected DORIS VTEC values, TECU |
| `window_size` | Selected spatial search radius, km |
| `cycle_number`, `pass_in_cycle` | Cycle and pass identifiers |

Unmatched points retain NaN interpolated VTEC, zero count and NaN window
size. The spread in `doris_points_std` is a neighbor statistic, not a
validated prediction uncertainty. Output rows are sorted by `obs_epoch`.

## Time identity and cycle boundaries

All DORIS–altimetry matching uses **TAI against TAI**. The main pipeline
calls `load_doris_cycle(..., require_corrected_tai=True)` and requires
the DORIS product attributes:

```python
obs_epoch_time_scale = "TAI"
obs_epoch_receiver_clock_corrected = True
```

Rebuild older DORIS products through Stage 1 and Stage 2 before using
them in the main pipeline. Direct reader calls default to
`require_corrected_tai=False`, allowing historical products to be read
with `legacy_unspecified` identity. They do not upgrade unknown times
to corrected TAI. Mixing unknown-time and corrected-TAI products is rejected.

The main pipeline reads DORIS observations **300 seconds before and
after the cycle** to supply its existing pass window. This expands both
file discovery and row selection. Direct calls to
`load_doris_cycle` retain the cycle-only default (`time_margin_sec=0`).
Altimetry targets remain within the original inclusive cycle boundaries.

## Processing steps and unchanged settings

1. Load cycle data. Missing daily files are skipped; no available data
   for a required stream raises an error.
2. Convert cycle start from TAI to UTC to select and propagate one SINEX
   solution per active station for the cycle-level spatial prefilter.
3. Apply the existing station-network prefilter. It tests the sum of
   Cartesian distances to the **two nearest stations** against
   `2 * max_station_distance * 1000` in metres.
4. Interpolate GIM at the retained altimetry IPPs, converting observation
   TAI to UTC for map queries and daily-file selection.
5. For each altimetry pass, preselect DORIS within the inclusive interval
   `[pass_start - 300 s, pass_end + 300 s]`.
6. For each target, retain DORIS with `elevation > elev_thres` and an
   absolute time difference **strictly less than 400 seconds**. Apply
   the existing Earth-rotation longitude correction relative to that target.
7. Expand the spatial radius until enough candidates are found, then
   calculate IDW. Collect pass results and save the cycle table.

Configuration is in [`config.py`](config.py):

| Setting | Default / role |
|---|---|
| `min_obs_count` | 30 candidates for temporal and spatial acceptance |
| `elev_thres` | 10 degrees; strict `>` mask |
| `min_distance` | 100 km initial search radius |
| `max_distance` | 1000 km maximum search radius |
| `distance_step` | 100 km radius increment |
| `max_station_distance` | 167500 km; the existing, very loose cycle prefilter setting |

The 300-second pass window and 400-second point window are separate
existing limits; the cycle halo does not enlarge either. Passes use
`ProcessPoolExecutor`; worker exceptions are reported and those passes
are skipped. A nonempty pass can still contain unmatched point rows.

`range_ratio_ja3` and `enable_roti` are present in the configuration but
are not consumed by the current interpolation path. DORIS VTEC has
already been leveled upstream; INPO applies no second leveling ratio.

## GIM and IDW boundary behavior

GIM interpolation is Earth-fixed bilinear spatial interpolation plus
linear temporal interpolation. Multi-day loading checks grids and the
declared IONEX `INTERVAL`, supplementing neighboring UTC days when needed
for endpoint support. Missing maps or a temporal gap larger than
`INTERVAL` yield NaN for unsupported queries. Exact map epochs remain
usable. Inconsistent grids/intervals and non-midnight map conflicts
raise errors. Conflicting shared-midnight maps use the file belonging
to **that UTC day**, independently of load order.

IDW uses the original `1 / distance` weights for positive finite
distances and values. If any finite values are at exactly zero distance,
their mean is returned. If no valid weights exist, IDW returns NaN.
This fixes coincident-point handling without replacing the interpolation
method, spatial radii or thresholds.

`gim_vtec` is stored as the original GIM value. Comparisons against DORIS
or altimetry should state the reference population and any externally
applied scaling; the saved column itself is not a scaled reference.
