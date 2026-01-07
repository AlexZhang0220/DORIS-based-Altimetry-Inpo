DORIS-based Altimetry VTEC Interpolation Pipeline
================================================

This document describes the interpolation pipeline that transfers
DORIS-based VTEC estimates onto altimetry observation points. The focus is on
the processing flow and module responsibilities, rather than scientific
interpretation.

----------------------------------------------------------------------
1. Pipeline Overview
----------------------------------------------------------------------

The pipeline operates at the altimetry cycle level and consists of four
main stages:

1. Cycle-level data loading
2. Station coordinate propagation and spatial prefiltering
3. GIM reference VTEC interpolation
4. Pass-level DORIS-to-altimetry interpolation (parallelized)

Each cycle is processed independently, while individual altimetry passes
within a cycle are processed in parallel.

----------------------------------------------------------------------
2. Input Data
----------------------------------------------------------------------

2.1 Altimetry VTEC

Altimetry VTEC data are loaded using load_altimetry_cycle() within the
cycle time window.

The data contain:
- Observation epoch (obs_epoch)
- Altimetry IPP latitude and longitude
- Pass index within the cycle (pass_in_cycle)
- Altimetry-based VTEC information

2.2 DORIS VTEC

DORIS VTEC data are loaded using load_doris_cycle() within the same cycle
time window.

The data contain:
- DORIS station code
- Observation epoch
- DORIS-based VTEC estimates
- Station-related metadata

2.3 Global Ionospheric Map (GIM)

Global Ionospheric Map data are loaded once per cycle using
load_gim_cycle().

The function returns:
- Time-indexed GIM VTEC maps
- Latitude grid
- Longitude grid

These data are used to compute reference VTEC values at altimetry IPP
locations.

2.4 Station Coordinates (SINEX)

Station coordinates and velocities are read from a SINEX file using
read_sinex_file().

For each station, position and velocity parameters are available together
with their validity time span.

----------------------------------------------------------------------
3. Cycle-level Preprocessing
----------------------------------------------------------------------

3.1 Station Coordinate Propagation

For each DORIS station appearing in the cycle:

- The SINEX solution valid at the cycle start epoch is selected
- Station coordinates are propagated from the reference epoch to the
  cycle start epoch using linear velocity models

The resulting station Cartesian coordinates are stored in a
cycle-specific table.

3.2 Distance-based Prefiltering of Altimetry Points

Altimetry IPPs are spatially prefiltered using altimetry_distance_mask().

For each altimetry point, the distance to all active DORIS stations is
computed. Altimetry points without any station within the maximum allowed
distance are discarded.

The distance threshold is defined in DEFAULT_INPO_CONFIG.

3.3 GIM VTEC Interpolation

Reference GIM VTEC values are interpolated onto all remaining altimetry
points using get_gim_vtec().

This step includes:
- Temporal interpolation between adjacent GIM epochs
- Spatial interpolation on the GIM latitude and longitude grids

The interpolated GIM VTEC is appended to the altimetry DataFrame as a new
column.

----------------------------------------------------------------------
4. Pass-level Interpolation
----------------------------------------------------------------------

Each altimetry cycle is divided into independent passes, identified by
pass_in_cycle. Pass-level processing is parallelized.

4.1 Time Window Selection

For each altimetry pass:

- The minimum and maximum observation epochs are determined
- DORIS observations are selected within an extended time window:
  [t_start - 5 minutes, t_end + 5 minutes]

If no DORIS observations fall into this window, the pass is skipped.

4.2 Core Interpolation

The core interpolation is performed using altimetry_inpo().

Inputs:
- Altimetry observations of a single pass
- Corresponding DORIS observations
- Interpolation and filtering settings

The function performs spatio-temporal matching and interpolation to
estimate DORIS-based VTEC values at altimetry IPP locations.

If the interpolation result is empty, the pass is discarded.

4.3 Bookkeeping

For each successfully processed pass:
- cycle_number is added
- pass_in_cycle is added

All valid pass-level results are collected for cycle-level output.

----------------------------------------------------------------------
5. Output Products
----------------------------------------------------------------------

After all passes of a cycle are processed:

- Pass-level results are concatenated
- Observations are sorted by observation epoch
- The final DataFrame is saved as a pickle file

Output path format:
DBAVStorage/<satellite>/cycle_unit/cycleXXX.pkl

Each output file contains:
- Altimetry observation time and IPP location
- Interpolated DORIS-based VTEC
- Reference GIM VTEC
- Cycle number and pass identifier

----------------------------------------------------------------------
6. Parallelization Strategy
----------------------------------------------------------------------

Parallelization is applied only at the pass level.

- Cycle-level data are loaded once per cycle
- Passes are processed independently using ProcessPoolExecutor
- The number of workers is limited to avoid oversubscription

This design balances computational efficiency and memory usage.

----------------------------------------------------------------------
7. Configuration
----------------------------------------------------------------------

All interpolation and filtering parameters are defined in
DEFAULT_INPO_CONFIG.

This includes:
- Maximum station distance for prefiltering
- Interpolation window settings
- Other algorithmic thresholds

----------------------------------------------------------------------
8. Execution
----------------------------------------------------------------------

The pipeline is executed via the main script:

python main.py

Cycle selection, satellite identifier, and input/output paths are defined
in the configuration section of the script.