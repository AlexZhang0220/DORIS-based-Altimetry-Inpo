"""Regression checks for missing phase and station solution interval repairs."""

import numpy as np
import pandas as pd
import pytest

import constant
from doris_vtec.io.utils import correct_receiver_epochs
from doris_vtec.stage1_obs2stec.geometry.station import (
    SECONDS_PER_YEAR,
    add_station_position,
)
from doris_vtec.stage2_stec2vtec.pass_detection.detect_passes import detect_passes
from doris_vtec.stage2_stec2vtec.pass_detection.doris_cycle_slip import (
    _fit_residuals,
    split_by_cycle_slip,
)


def _phase_observations(count=80, slip=False):
    """Zenith geometry with a known geometric phase and optional L2 jump."""
    rho = 1_336_000.0 + 2.0 * np.arange(count)
    frame = pd.DataFrame({
        "row_id": np.arange(count),
        "obs_epoch": pd.date_range("2018-01-01", periods=count, freq="3s"),
        "station_code": "TEST", "prn": "L39", "antenna_type": "STAREC B",
        "station_freq_shift": 0, "elevation": 90.0,
        "sta_x": constant.AE84, "sta_y": 0.0, "sta_z": 0.0,
        "sat_x": constant.AE84 + rho, "sat_y": 0.0, "sat_z": 0.0,
        "L1": (rho - constant.jason3_L1_pco - constant.starec_L1_pco)
              / constant.d_lambda1,
        "L2": (rho - constant.jason3_L2_pco - constant.starec_L2_pco)
              / constant.d_lambda2,
    })
    if slip:
        frame.loc[40:, "L2"] += 100.0
    return frame


def _pass_lengths(frame):
    return frame.groupby("pass_id", sort=False).size().tolist()


@pytest.mark.parametrize("slip,expected_lengths", [(False, [80]), (True, [40, 40])])
def test_complete_phase_preserves_original_pass_rows(slip, expected_lengths):
    frame = _phase_observations(slip=slip)
    result = detect_passes(frame, min_obs_count=30, elev_thres=10)
    assert _pass_lengths(result) == expected_lengths
    np.testing.assert_array_equal(result.row_id, frame.row_id)


@pytest.mark.parametrize("column,bad_value", [("L1", np.nan), ("L2", np.inf)])
def test_bad_phase_no_longer_hides_known_cycle_slip(column, bad_value):
    frame = _phase_observations(slip=True)
    frame.loc[10, column] = bad_value
    result = detect_passes(frame, min_obs_count=30, elev_thres=10)
    assert _pass_lengths(result) == [39, 40]
    np.testing.assert_array_equal(result.row_id, np.delete(np.arange(80), 10))


def test_missing_phases_are_removed_before_existing_nine_second_gap_rule():
    frame = _phase_observations(count=100)
    frame.loc[40:44, "L1"] = np.nan
    result = detect_passes(frame, min_obs_count=30, elev_thres=10)
    assert _pass_lengths(result) == [40, 55]
    np.testing.assert_array_equal(result.row_id, np.r_[0:40, 45:100])


def test_all_missing_phases_produces_no_pass():
    frame = _phase_observations()
    frame["L1"] = np.nan
    assert detect_passes(frame, min_obs_count=30, elev_thres=10).empty


def test_missing_geometry_cannot_return_an_apparently_clean_pass():
    frame = _phase_observations()
    frame.loc[10, "sta_x"] = np.nan
    with pytest.raises(ValueError, match="Non-finite cycle-slip fit input"):
        detect_passes(frame, min_obs_count=30, elev_thres=10)


@pytest.mark.parametrize("invalid_part", ["X", "y"])
def test_residual_fit_rejects_nonfinite_input_before_lstsq(monkeypatch, invalid_part):
    X = np.ones((3, 1))
    y = np.zeros(3)
    if invalid_part == "X":
        X[0, 0] = np.nan
    else:
        y[0] = np.inf

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Invalid input reached least-squares fit")

    monkeypatch.setattr(np.linalg, "lstsq", unexpected_fit)
    with pytest.raises(ValueError, match="Non-finite cycle-slip fit input"):
        _fit_residuals(X, y)


def test_both_detector_fits_validate_residuals(monkeypatch):
    real_fit = np.linalg.lstsq
    call_count = 0

    def second_fit_is_nonfinite(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = real_fit(*args, **kwargs)
        if call_count == 2:
            return (np.full_like(result[0], np.nan), *result[1:])
        return result

    monkeypatch.setattr(np.linalg, "lstsq", second_fit_is_nonfinite)
    with pytest.raises(ValueError, match="Non-finite cycle-slip fit residuals"):
        split_by_cycle_slip(_phase_observations(), min_obs_count=30)
    assert call_count == 2


def test_guard_does_not_change_finite_least_squares_result():
    X = np.column_stack((np.ones(5), np.arange(5)))
    y = pd.Series([2.1, 3.0, 4.2, 5.0, 6.1])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pd.testing.assert_series_equal(_fit_residuals(X, y), y - X @ beta)


def _station_solution(start="2018-01-01", end="2018-01-01 00:00:09", **changes):
    record = dict(
        station_code="TEST", antenna_type="STAREC B",
        start_obs=pd.Timestamp(start), end_obs=pd.Timestamp(end),
        ref_epoch=pd.Timestamp("2017-01-01"),
        x=6_000_000.0, y=100_000.0, z=2_000_000.0,
        vx=0.25, vy=-0.125, vz=0.5,
    )
    record.update(changes)
    return record


def test_station_switches_solution_without_per_epoch_velocity_propagation():
    observations = pd.DataFrame({
        "row_id": [2, 0, 3, 1], "station_code": "TEST",
        "obs_epoch": pd.to_datetime([
            "2018-01-01 00:00:12", "2018-01-01 00:00:00",
            "2018-01-01 00:00:15", "2018-01-01 00:00:03",
        ]),
    }, index=[5, 5, 9, 9])
    observations.attrs.update(test_evidence="kept")
    solutions = pd.DataFrame([
        _station_solution(),
        _station_solution("2018-01-01 00:00:10", "2018-01-02",
                          antenna_type="ALCATEL", x=6_000_010.0),
    ])
    original = observations.copy()
    result = add_station_position(observations, solutions)
    assert result.row_id.tolist() == [2, 0, 3, 1]
    assert result.antenna_type.tolist() == ["ALCATEL", "STAREC B", "ALCATEL", "STAREC B"]
    for rows, sol_idx, ref_obs in [([0, 2], 1, 0), ([1, 3], 0, 1)]:
        sol = solutions.iloc[sol_idx]
        dt = (observations.obs_epoch.iloc[ref_obs] - sol.ref_epoch).total_seconds()
        for axis in "xyz":
            expected = sol[axis] + sol[f"v{axis}"] * (dt / SECONDS_PER_YEAR)
            np.testing.assert_array_equal(result.loc[rows, f"sta_{axis}"], [expected] * 2)
    assert result.attrs == observations.attrs
    pd.testing.assert_frame_equal(observations, original)


def test_one_station_solution_preserves_original_numeric_result_and_columns():
    observations = pd.DataFrame({
        "station_code": ["TEST", "TEST"],
        "obs_epoch": pd.to_datetime(["2018-01-01 00:00:03", "2018-01-01 00:00:06"]),
    })
    solutions = pd.DataFrame([_station_solution()])
    sol = solutions.iloc[0]
    dt_year = (observations.obs_epoch.iloc[0] - sol.ref_epoch).total_seconds() / SECONDS_PER_YEAR
    old_coordinates = pd.DataFrame([{
        "station_code": "TEST", "antenna_type": sol.antenna_type,
        "sta_x": sol.x + sol.vx * dt_year,
        "sta_y": sol.y + sol.vy * dt_year,
        "sta_z": sol.z + sol.vz * dt_year,
    }])
    expected = observations.merge(old_coordinates, on="station_code", how="left")
    pd.testing.assert_frame_equal(add_station_position(observations, solutions), expected,
                                  check_exact=True)


@pytest.mark.parametrize("empty_solutions", [False, True])
def test_station_no_matching_solution_retains_nan_and_empty_antenna(empty_solutions):
    observations = pd.DataFrame({
        "station_code": ["TEST", "OTHER"],
        "obs_epoch": pd.to_datetime(["2019-01-01", "2018-01-01"]),
    })
    solutions = pd.DataFrame([_station_solution()])
    if empty_solutions:
        solutions = solutions.iloc[:0]
    result = add_station_position(observations, solutions)
    assert len(result) == 2
    assert result[["sta_x", "sta_y", "sta_z"]].isna().all().all()
    assert result.antenna_type.tolist() == ["", ""]


def test_station_closed_interval_boundary_overlap_is_rejected():
    observations = pd.DataFrame({"station_code": ["TEST"],
                                 "obs_epoch": [pd.Timestamp("2018-01-01 00:00:09")]})
    solutions = pd.DataFrame([
        _station_solution(),
        _station_solution("2018-01-01 00:00:09", "2018-01-02"),
    ])
    with pytest.raises(ValueError, match="Overlapping SINEX solutions: TEST"):
        add_station_position(observations, solutions)


def test_station_only_rejects_overlap_where_an_observation_has_two_solutions():
    observations = pd.DataFrame({"station_code": ["TEST"],
                                 "obs_epoch": [pd.Timestamp("2018-01-01 00:00:00")]})
    solutions = pd.DataFrame([
        _station_solution(),
        _station_solution("2018-01-01 00:00:09", "2018-01-02"),
    ])
    assert np.isfinite(add_station_position(observations, solutions).sta_x.iloc[0])


def test_station_corrected_tai_uses_utc_interval_and_preserves_epoch_attrs():
    raw = pd.DataFrame({"station_code": ["TEST", "TEST"],
                        "obs_epoch": pd.to_datetime([
                            "2018-01-01 00:00:36", "2018-01-01 00:00:38"]),
                        "receiver_clock_offset": 0.0})
    corrected = correct_receiver_epochs(raw)
    solutions = pd.DataFrame([
        _station_solution("2017-12-31 00:00:00", "2017-12-31 23:59:59",
                          antenna_type="STAREC B"),
        _station_solution("2018-01-01 00:00:00", "2018-01-01 00:00:09",
                          antenna_type="ALCATEL"),
    ])
    result = add_station_position(corrected, solutions)
    assert result.antenna_type.tolist() == ["STAREC B", "ALCATEL"]
    pd.testing.assert_series_equal(result.obs_epoch, corrected.obs_epoch)
    assert result.attrs == corrected.attrs


def test_station_missing_validity_boundary_is_not_inferred_as_infinite():
    observations = pd.DataFrame({"station_code": ["TEST"],
                                 "obs_epoch": [pd.Timestamp("2018-01-01")]})
    solutions = pd.DataFrame([_station_solution(start=pd.NaT)])
    result = add_station_position(observations, solutions)
    assert np.isnan(result.sta_x.iloc[0])
    assert result.antenna_type.iloc[0] == ""
