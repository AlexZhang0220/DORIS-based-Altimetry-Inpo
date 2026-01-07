DORIS-based VTEC Processing Pipeline
===================================

This repository implements a complete processing pipeline that converts
raw DORIS observation data into leveled Vertical Total Electron Content
(VTEC) products and interpolates them onto altimetry observation points.
The documentation focuses strictly on software workflow and data flow.

----------------------------------------------------------------------
1. Pipeline Overview
----------------------------------------------------------------------

The processing chain is organized into two main stages:

- Stage 1: Raw DORIS observations to Slant TEC (STEC)
- Stage 2: STEC to leveled VTEC and altimetry interpolation

Stage 1 and Stage 2 are independent. Intermediate products generated in
Stage 1 are stored on disk and reused by Stage 2 without re-reading raw
observation files.

----------------------------------------------------------------------
2. Stage 1: DORIS Observation to STEC
----------------------------------------------------------------------

Stage 1 converts raw DORIS RINEX observation files into raw Slant Total
Electron Content (STEC) time series.

The main processing steps are:

1. DORIS RINEX observation files are read and parsed into structured
   tables containing observation epochs, station identifiers, satellite
   identifiers, and dual-frequency phase measurements.

2. Station coordinates are read from SINEX files, and precise satellite
   positions are read from SP3 orbit files. These external data are
   interpolated to the observation epochs as required.

3. Geometric parameters are computed for each observation, including
   satellite–station line-of-sight vectors, elevation angles, and
   ionospheric pierce point (IPP) coordinates.

4. Raw STEC values are computed from dual-frequency phase observations.
   The resulting STEC time series represent relative ionospheric
   variations and include an unknown constant bias.

The outputs of Stage 1 are intermediate STEC products stored in a
structured tabular format.

----------------------------------------------------------------------
3. Stage 2: STEC to Leveled VTEC
----------------------------------------------------------------------

Stage 2 converts intermediate STEC products into leveled Vertical Total
Electron Content (VTEC) products using external ionospheric references.

The processing steps are:

1. Intermediate STEC products are read from disk and grouped by station
   and satellite to form continuous observation passes. Each pass
   represents a temporally continuous tracking arc with consistent
   integer ambiguity.

2. For each pass, a reference epoch is selected based on the maximum
   elevation angle. At this epoch, reference VTEC values are obtained
   from Global Ionospheric Map (GIM) products by interpolating the GIM
   data in time and space to the corresponding IPP.

3. A pass-wise leveling procedure is applied. A constant offset is
   determined at the reference epoch by aligning the STEC-derived
   ionospheric delay with the GIM-based TEC reference. This offset is
   applied to all observations within the same pass.

4. The leveled STEC values are converted to VTEC using the geometric
   mapping function, and the leveled VTEC products are written to disk.

----------------------------------------------------------------------
4. Altimetry VTEC Interpolation
----------------------------------------------------------------------

After leveled DORIS-based VTEC products are available, they are
interpolated onto altimetry observation points on a cycle-by-cycle
basis.

The interpolation workflow is:

1. Altimetry VTEC data and DORIS-based VTEC data are loaded within the
   same altimetry cycle time window.

2. Station coordinates are propagated to the cycle reference epoch
   using SINEX position and velocity parameters.

3. Altimetry ionospheric pierce points are spatially prefiltered based
   on their distance to active DORIS stations to reduce unnecessary
   interpolation.

4. Reference GIM VTEC values are interpolated to all remaining altimetry
   IPPs for comparison and validation purposes.

5. Each altimetry cycle is divided into independent passes. For each
   pass, DORIS observations within a local time window are selected and
   interpolated to altimetry IPPs using spatio-temporal matching.

6. Pass-level interpolation is parallelized to improve computational
   efficiency. Valid pass results are concatenated and stored as
   cycle-level interpolation products.

----------------------------------------------------------------------
5. Input and Output Summary
----------------------------------------------------------------------

Stage 1 Inputs:
- DORIS RINEX observation files
- SP3 precise orbit files
- SINEX station coordinate files

Stage 1 Outputs:
- Intermediate STEC products

Stage 2 Inputs:
- Intermediate STEC products
- Global Ionospheric Map (GIM) files

Stage 2 Outputs:
- Leveled DORIS-based VTEC products

Interpolation Inputs:
- Leveled DORIS-based VTEC products
- Altimetry VTEC data
- Global Ionospheric Map (GIM) files

Interpolation Outputs:
- DORIS-based VTEC interpolated to altimetry IPPs
- Cycle- and pass-level interpolation products

----------------------------------------------------------------------
6. Execution and Configuration
----------------------------------------------------------------------

Each processing stage can be executed independently. Interpolation and
filtering parameters are centralized in configuration files to ensure
reproducibility and consistent processing across cycles.
