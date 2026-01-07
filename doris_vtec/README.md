DORIS-based VTEC Processing Pipeline
===================================

This document describes the complete DORIS-based VTEC processing
pipeline, which converts raw DORIS observation files into leveled
Vertical Total Electron Content (VTEC) products. The focus is strictly
on the software workflow and data flow between processing stages.

----------------------------------------------------------------------
1. Purpose
----------------------------------------------------------------------

This repository implements a two-stage processing pipeline for DORIS
data. The pipeline transforms raw DORIS observations into scientifically
usable VTEC products through intermediate processing steps.

The documentation focuses exclusively on:
- Processing logic
- Data dependencies
- Input and output relationships

Scientific interpretation and analysis are intentionally excluded.

----------------------------------------------------------------------
2. Overall Workflow
----------------------------------------------------------------------

The processing pipeline is organized into two sequential and independent
stages:

- Stage 1: Raw DORIS observations to Slant TEC (STEC)
- Stage 2: STEC to leveled Vertical TEC (VTEC)

Stage 1 and Stage 2 can be executed independently. The intermediate
outputs produced by Stage 1 are stored on disk and reused by Stage 2,
eliminating the need to re-read or re-process the original observation
files.

----------------------------------------------------------------------
3. Stage 1: DORIS Observation to STEC
----------------------------------------------------------------------

Stage 1 is responsible for converting raw DORIS observation files into
raw Slant Total Electron Content (STEC) time series.

The processing steps in Stage 1 are as follows.

3.1 Observation File Reading

DORIS RINEX observation files are read and parsed into structured data
tables. The following information is extracted for each observation
epoch:
- Observation time
- Station identifier
- Satellite identifier
- Dual-frequency phase measurements

3.2 External Data Loading

Station coordinates are read from SINEX files, including position and
velocity parameters. Precise satellite orbits are read from SP3 files.

Both station coordinates and satellite positions are interpolated to
the exact observation epochs when required.

3.3 Geometric Parameter Computation

Using the station and satellite positions, geometric parameters are
computed for each observation, including:
- Satellite–station line-of-sight vectors
- Elevation angles
- Ionospheric pierce point (IPP) coordinates

These parameters are appended to the observation data records.

3.4 STEC Computation

Raw Slant TEC values are computed from the dual-frequency phase
observations. The resulting STEC time series represent relative
ionospheric variations and include an unknown constant bias.

3.5 Stage 1 Outputs

The outputs of Stage 1 consist of intermediate STEC products stored on
disk in a structured tabular format. These products serve as the sole
input to Stage 2.

----------------------------------------------------------------------
4. Stage 2: STEC to Leveled VTEC
----------------------------------------------------------------------

Stage 2 converts the intermediate STEC products generated in Stage 1
into leveled Vertical Total Electron Content (VTEC) products.

The processing steps in Stage 2 are summarized below.

4.1 Pass Segmentation

Intermediate STEC products are read from disk and grouped by station
and satellite. The data are segmented into continuous observation
passes.

Each pass represents a temporally continuous tracking arc with
consistent integer ambiguity.

4.2 Reference Epoch Selection and GIM Interpolation

For each pass, a reference epoch is selected based on the maximum
elevation angle within the pass.

At the reference epoch, external reference VTEC values are obtained
from Global Ionospheric Map (GIM) products. The GIM data are interpolated
in both time and space to the ionospheric pierce point corresponding to
the reference epoch.

4.3 Pass-wise Leveling

A pass-wise leveling procedure is applied by determining a constant
offset at the reference epoch. This offset aligns the STEC-derived
ionospheric delay with the GIM-based TEC reference.

The determined offset is applied uniformly to all observations within
the same pass.

4.4 VTEC Conversion and Output

After leveling, the corrected STEC values are converted to VTEC using
the geometric mapping function.

The final leveled VTEC products, together with associated geometric
parameters, are written to disk.

----------------------------------------------------------------------
5. Input and Output Summary
----------------------------------------------------------------------

Stage 1 Inputs:
- DORIS RINEX observation files
- SP3 precise orbit files
- SINEX station coordinate files

Stage 1 Outputs:
- Intermediate STEC products in structured tabular format

Stage 2 Inputs:
- Intermediate STEC products generated by Stage 1
- Global Ionospheric Map (GIM) files

Stage 2 Outputs:
- Leveled VTEC products
- Associated geometric and pass-level parameters