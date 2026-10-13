"""Independent endpoint geometry checks for the small GitHub PCO fix."""

import numpy as np
import pandas as pd
import pytest

import constant
from doris_vtec.stage1_obs2stec.geometry.elevation import (
    add_elevation,
    compute_pco_projection_cosines,
)
from doris_vtec.stage1_obs2stec.ionosphere.dion import add_doris_stec
from doris_vtec.stage2_stec2vtec.pass_detection.doris_cycle_slip import (
    _projected_pco_phase_background,
    split_by_cycle_slip,
)


def _ecef(latitude, longitude, height):
    """Independent WGS84 forward geometry; no production transform call."""
    lat, lon = np.deg2rad([latitude, longitude])
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    n = 6378137.0 / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array([
        (n + height) * np.cos(lat) * np.cos(lon),
        (n + height) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - e2) + height) * np.sin(lat),
    ])


def _normal(latitude, longitude):
    lat, lon = np.deg2rad([latitude, longitude])
    return np.array([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon), np.sin(lat)])


def _observation(station_geo=(45.0, 10.0, 100.0),
                 satellite_geo=(47.0, 20.0, 1336000.0), antenna="STAREC B"):
    station = _ecef(*station_geo)
    satellite = _ecef(*satellite_geo)
    frame = pd.DataFrame([dict(zip(
        ["sta_x", "sta_y", "sta_z", "sat_x", "sat_y", "sat_z"],
        np.concatenate((station, satellite)),
    ))])
    frame["L1"] = 0.0
    frame["L2"] = 0.0
    frame["station_freq_shift"] = 0
    frame["antenna_type"] = antenna
    frame["prn"] = "L39"
    return add_elevation(frame)


def test_projection_uses_two_independent_ellipsoidal_endpoint_normals():
    station_geo = (45.0, 10.0, 100.0)
    satellite_geo = (47.0, 20.0, 1336000.0)
    frame = _observation(station_geo, satellite_geo)
    los = _ecef(*satellite_geo) - _ecef(*station_geo)
    los /= np.linalg.norm(los)
    expected_satellite = np.dot(_normal(*satellite_geo[:2]), los)
    expected_ground = np.dot(_normal(*station_geo[:2]), los)
    satellite, ground = compute_pco_projection_cosines(frame)
    np.testing.assert_allclose(satellite, expected_satellite, atol=2e-9, rtol=0)
    np.testing.assert_allclose(ground, expected_ground, atol=2e-9, rtol=0)
    np.testing.assert_allclose(ground, np.sin(np.deg2rad(frame.elevation)),
                               atol=1e-14, rtol=0)
    assert abs(satellite[0] - ground[0]) > 0.02


@pytest.mark.parametrize("antenna,ground_l1,ground_l2", [
    ("STAREC B", constant.starec_L1_pco, constant.starec_L2_pco),
    ("ALCATEL", constant.alcatel_L1_pco, constant.alcatel_L2_pco),
])
def test_stec_and_detector_use_same_signed_frequency_endpoint_corrections(
        antenna, ground_l1, ground_l2):
    frame = _observation(antenna=antenna)
    satellite, ground = compute_pco_projection_cosines(frame)
    correction_l1 = constant.jason3_L1_pco * satellite + ground_l1 * ground
    correction_l2 = constant.jason3_L2_pco * satellite + ground_l2 * ground
    wavelength_l1 = constant.c / constant.d_f1
    wavelength_l2 = constant.c / constant.d_f2
    k = constant.d_f2 ** 2 / (constant.d_f1 ** 2 - constant.d_f2 ** 2)
    result = add_doris_stec(frame.copy())
    np.testing.assert_allclose(result.dion, k * (correction_l1 - correction_l2),
                               atol=1e-14, rtol=0)
    phase_background = _projected_pco_phase_background(
        frame, wavelength_l1, wavelength_l2, ground_l1, ground_l2
    )
    np.testing.assert_allclose(
        phase_background,
        5.0 * correction_l1 / wavelength_l1 - correction_l2 / wavelength_l2,
        atol=1e-14, rtol=0,
    )


def test_zenith_keeps_old_pco_result():
    frame = _observation((0.0, 0.0, 0.0), (0.0, 0.0, 1336000.0))
    satellite, ground = compute_pco_projection_cosines(frame)
    np.testing.assert_allclose([satellite[0], ground[0]], [1.0, 1.0],
                               atol=1e-15, rtol=0)
    result = add_doris_stec(frame.copy())
    expected_old = constant.d_k * (constant.d_geomcorrJa3 + constant.d_STAREC)
    np.testing.assert_allclose(result.dion, expected_old, atol=1e-14, rtol=0)


def test_off_nadir_changes_only_satellite_pco_projection():
    frame = _observation()
    result = add_doris_stec(frame.copy())
    satellite, ground = compute_pco_projection_cosines(frame)
    old_correction = constant.d_k * (
        constant.d_geomcorrJa3 + constant.d_STAREC
    ) * np.sin(np.deg2rad(frame.elevation.to_numpy()))
    np.testing.assert_allclose(
        result.dion.to_numpy() - old_correction,
        constant.d_k * constant.d_geomcorrJa3 * (satellite - ground),
        atol=1e-14, rtol=0,
    )
    assert abs(result.dion.iloc[0] - old_correction[0]) > 1e-5


@pytest.mark.parametrize("slip", [False, True])
def test_existing_two_stage_detector_keeps_smooth_arc_and_splits_phase_jump(slip):
    frame = pd.concat([
        _observation(satellite_geo=(47.0, longitude, 1336000.0))
        for longitude in np.linspace(17.0, 21.0, 80)
    ], ignore_index=True)
    frame["obs_epoch"] = pd.date_range("2018-01-01", periods=80, freq="3s")
    satellite, ground = compute_pco_projection_cosines(frame)
    rho = np.linalg.norm(frame[["sat_x", "sat_y", "sat_z"]].to_numpy()
                         - frame[["sta_x", "sta_y", "sta_z"]].to_numpy(), axis=1)
    correction_l1 = constant.jason3_L1_pco * satellite + constant.starec_L1_pco * ground
    correction_l2 = constant.jason3_L2_pco * satellite + constant.starec_L2_pco * ground
    frame["L1"] = (rho - correction_l1) / constant.d_lambda1
    frame["L2"] = (rho - correction_l2) / constant.d_lambda2
    if slip:
        frame.loc[40:, "L2"] += 100.0
    segments = split_by_cycle_slip(frame, min_obs_count=30)
    assert [len(segment) for segment in segments] == ([40, 40] if slip else [80])


def test_degenerate_geometry_retains_missing_values_for_existing_elevation_filter():
    frame = _observation()
    frame.loc[:, ["sat_x", "sat_y", "sat_z"]] = frame[
        ["sta_x", "sta_y", "sta_z"]
    ].to_numpy()
    satellite, ground = compute_pco_projection_cosines(frame)
    assert np.isnan(satellite).all() and np.isnan(ground).all()


def test_missing_station_does_not_abort_other_valid_observations():
    frame = pd.concat([_observation(), _observation()], ignore_index=True)
    frame.loc[1, "sta_x"] = np.nan
    satellite, ground = compute_pco_projection_cosines(frame)
    assert np.isfinite(satellite[0]) and np.isfinite(ground[0])
    assert np.isnan(satellite[1]) and np.isnan(ground[1])
    result = add_doris_stec(frame)
    assert np.isfinite(result.stec.iloc[0]) and np.isnan(result.stec.iloc[1])
