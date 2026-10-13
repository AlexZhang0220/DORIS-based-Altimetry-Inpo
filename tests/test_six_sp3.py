"""SP3 identity, support and duplicate-state regressions for the six fixes."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from doris_vtec.io.sp3_reader import _sp3_dataset_to_dataframe
from doris_vtec.stage1_obs2stec.geometry.satellite import add_satellite_position


START = pd.Timestamp("2018-01-01")


def _orbit(prn="L39"):
    i = np.arange(14, dtype=float)
    return pd.DataFrame({
        "epoch": START + pd.to_timedelta(i * 30, unit="s"), "prn": prn,
        "x": 7e6 + 1e4 * np.sin(.7 * i),
        "y": 1e6 + 2e4 * np.cos(.4 * i),
        "z": 2e6 + 3e4 * np.sin(.3 * i),
    })


def _observations(seconds, prn="L39"):
    return pd.DataFrame({
        "obs_epoch": START + pd.to_timedelta(seconds, unit="s"),
        "receiver_clock_offset": 0.0, "prn": prn,
    })


def _dataset(positions=None, velocities=None):
    if positions is None:
        positions = [[7000.0, 1000.0, 2000.0], [7000.0, 1000.0, 2000.0]]
    if velocities is None:
        velocities = [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]
    return xr.Dataset(
        {
            "position": (("time", "sv", "ECEF"), np.asarray(positions)[:, None, :]),
            "velocity": (("time", "sv", "ECEF"), np.asarray(velocities)[:, None, :]),
        },
        coords={"time": [START, START], "sv": ["L39"], "ECEF": ["x", "y", "z"]},
    )


def test_normal_eight_point_results_equal_frozen_legacy_results():
    # Generated from the frozen pre-six-fix function with this nonlinear orbit.
    expected = np.array([
        [7000000.0, 1020000.0, 2000000.0],
        [7003434.245899983, 1019601.3807867955, 2004483.1644141218],
        [7009161.273662392, 1008472.8562188293, 2022538.4131387384],
        [6990132.103349834, 982862.236489311, 2027868.7895376915],
        [7001330.759132122, 982694.268686885, 2011449.8294460871],
        [7006250.972215646, 1005673.1284492591, 1982853.1652182452],
        [7003190.983623493, 1009370.3334260075, 1979367.0152244808],
    ])
    result = add_satellite_position(_observations([0, 15, 85, 195, 275, 375, 390]), _orbit())
    np.testing.assert_array_equal(result[["sat_x", "sat_y", "sat_z"]].to_numpy(), expected)


def test_only_matching_prn_contributes_even_with_other_orbit_first():
    right = _orbit()
    wrong = _orbit("L40")
    wrong[["x", "y", "z"]] += 9e6
    mixed = pd.concat([wrong, right], ignore_index=True)
    obs = _observations([15, 195, 375])
    pd.testing.assert_frame_equal(add_satellite_position(obs, mixed), add_satellite_position(obs, right))


def test_wrong_prn_orbit_is_rejected():
    with pytest.raises(ValueError, match="No SP3 orbit for L39"):
        add_satellite_position(_observations([15]), _orbit("L40"))


@pytest.mark.parametrize("prns", [["L39", "L40"], ["L39", None], [None, None]])
def test_mixed_or_unknown_observation_prns_are_rejected(prns):
    obs = _observations([15, 45])
    obs["prn"] = prns
    with pytest.raises(ValueError, match="One observation PRN"):
        add_satellite_position(obs, _orbit())


@pytest.mark.parametrize("missing_from", ["observations", "orbit"])
def test_prn_column_is_required(missing_from):
    obs, orbit = _observations([15]), _orbit()
    if missing_from == "observations":
        obs = obs.drop(columns="prn")
    else:
        orbit = orbit.drop(columns="prn")
    with pytest.raises(ValueError, match="PRNs are required"):
        add_satellite_position(obs, orbit)


def test_outside_orbit_is_nan_but_exact_endpoints_are_valid():
    obs = _observations([-1, 0, 390, 391, 10000])
    result = add_satellite_position(obs, _orbit())
    coordinates = result[["sat_x", "sat_y", "sat_z"]]
    assert coordinates.iloc[[0, 3, 4]].isna().all().all()
    np.testing.assert_array_equal(coordinates.iloc[[1, 2]].to_numpy(), _orbit().iloc[[0, -1]][["x", "y", "z"]].to_numpy())
    assert result["obs_epoch"].tolist() == obs["obs_epoch"].tolist()


def test_support_sorting_does_not_change_positions():
    obs = _observations([375, 15, 195, 15])
    expected = add_satellite_position(obs, _orbit())
    shuffled = _orbit().sample(frac=1, random_state=19)
    pd.testing.assert_frame_equal(add_satellite_position(obs, shuffled), expected)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_orbit_coordinates_are_rejected(bad):
    orbit = _orbit()
    orbit.loc[3, "x"] = bad
    with pytest.raises(ValueError, match="Non-finite SP3 coordinates"):
        add_satellite_position(_observations([15]), orbit)


def test_missing_orbit_epoch_is_rejected():
    orbit = _orbit()
    orbit.loc[3, "epoch"] = pd.NaT
    with pytest.raises(ValueError, match="SP3 epochs must be finite"):
        add_satellite_position(_observations([15]), orbit)


def test_unparseable_orbit_epoch_is_rejected():
    orbit = _orbit()
    orbit["epoch"] = orbit["epoch"].astype(object)
    orbit.loc[3, "epoch"] = "invalid epoch"
    with pytest.raises(ValueError):
        add_satellite_position(_observations([15]), orbit)


def test_timezone_aware_orbit_epoch_is_rejected():
    orbit = _orbit()
    orbit["epoch"] = orbit["epoch"].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        add_satellite_position(_observations([15]), orbit)


def test_direct_dataframe_duplicate_epochs_are_rejected():
    orbit = _orbit()
    orbit = pd.concat([orbit, orbit.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate SP3 epochs"):
        add_satellite_position(_observations([15]), orbit)


def test_single_support_point_cannot_define_an_interpolation():
    result = add_satellite_position(_observations([0]), _orbit().iloc[:1])
    assert result[["sat_x", "sat_y", "sat_z"]].isna().all().all()


def test_empty_observations_do_not_require_orbit_identity():
    obs = pd.DataFrame({"obs_epoch": pd.Series(dtype="datetime64[ns]"),
                        "receiver_clock_offset": pd.Series(dtype=float)})
    result = add_satellite_position(obs, pd.DataFrame())
    assert result.empty
    assert {"sat_x", "sat_y", "sat_z"}.issubset(result.columns)
    assert result.attrs["obs_epoch_receiver_clock_corrected"] is True


def test_reader_identical_records_are_deduplicated_without_unit_changes():
    result = _sp3_dataset_to_dataframe(_dataset())
    assert len(result) == 1
    np.testing.assert_array_equal(result[["x", "y", "z"]].to_numpy(), [[7e6, 1e6, 2e6]])
    np.testing.assert_array_equal(result[["vx", "vy", "vz"]].to_numpy(), [[10., 20., 30.]])
    assert result["prn"].tolist() == ["L39"]


@pytest.mark.parametrize("variable", ["position", "velocity"])
def test_reader_nan_placeholder_does_not_conflict_with_a_finite_record(variable):
    ds = _dataset()
    ds[variable].values[0, :, :] = np.nan
    result = _sp3_dataset_to_dataframe(ds)
    expected = _sp3_dataset_to_dataframe(_dataset())
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("variable", ["position", "velocity"])
def test_reader_conflicting_records_raise_instead_of_averaging(variable):
    ds = _dataset()
    ds[variable].values[1, 0, 0] += 1e-9
    with pytest.raises(ValueError, match=f"Conflicting SP3 {variable}"):
        _sp3_dataset_to_dataframe(ds)


@pytest.mark.parametrize("variable", ["position", "velocity"])
def test_reader_rejects_infinite_state_records(variable):
    ds = _dataset()
    ds[variable].values[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match=f"Non-finite SP3 {variable}"):
        _sp3_dataset_to_dataframe(ds)


def test_reader_preserves_missing_velocity_as_nan():
    ds = _dataset()
    ds["velocity"].values[:] = np.nan
    result = _sp3_dataset_to_dataframe(ds)
    assert len(result) == 1
    assert result[["vx", "vy", "vz"]].isna().all().all()
