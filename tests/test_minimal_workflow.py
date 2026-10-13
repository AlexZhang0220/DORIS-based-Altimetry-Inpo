"""Clock identity and the existing Stage1/Stage2 flow with fixed external inputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import constant
from doris_vtec.io.utils import correct_receiver_epochs
from doris_vtec.io.ionex_reader import GimMaps
from doris_vtec.stage1_obs2stec.geometry.satellite import add_satellite_position
from doris_vtec.stage1_obs2stec.geometry.station import add_station_position
import doris_vtec.stage1_obs2stec.pipeline as stage1
import doris_vtec.stage2_stec2vtec.pipeline as stage2
from doris_vtec.config import DEFAULT_STAGE2_CONFIG
from doris_inpo.io.doris_vtec_reader import load_doris_cycle


@pytest.mark.parametrize("offset", [0.125, -0.125])
def test_receiver_offset_applied_once_across_pickle_roundtrip(tmp_path, offset):
    raw = pd.DataFrame({"obs_epoch": pd.to_datetime(["2018-01-01"]),
                        "receiver_clock_offset": [offset]})
    corrected = correct_receiver_epochs(raw)
    path = tmp_path / "epochs.pkl"
    corrected.to_pickle(path)
    again = correct_receiver_epochs(pd.read_pickle(path))
    pd.testing.assert_frame_equal(corrected, again)
    assert again.obs_epoch.iloc[0] == raw.obs_epoch.iloc[0] + pd.Timedelta(seconds=offset)
    assert raw.obs_epoch.iloc[0] == pd.Timestamp("2018-01-01")


def test_unknown_clock_is_not_invented():
    with pytest.raises(ValueError, match="offset is required"):
        correct_receiver_epochs(pd.DataFrame({"obs_epoch": [pd.Timestamp("2018-01-01")]}))


def test_orbit_uses_the_same_once_corrected_epoch_for_raw_and_marked_input():
    start = pd.Timestamp("2018-01-01")
    orbit = pd.DataFrame({"epoch": pd.date_range(start, periods=8, freq="1s"),
                          "x": 7_000_000.0 + 2.0 * np.arange(8), "y": 0.0, "z": 0.0,
                          "prn": "L39"})
    raw = pd.DataFrame({"obs_epoch": [start + pd.Timedelta(seconds=3)],
                        "receiver_clock_offset": [0.5], "prn": "L39"})
    a = add_satellite_position(raw, orbit)
    b = add_satellite_position(correct_receiver_epochs(raw), orbit)
    pd.testing.assert_frame_equal(a, b)
    assert a.sat_x.iloc[0] == pytest.approx(7_000_007.0, abs=1e-7)
    assert a.attrs["obs_epoch_receiver_clock_corrected"] is True


def _stations():
    return pd.DataFrame({
        "station_code": ["TEST"], "antenna_type": ["STAREC B"],
        "start_obs": [pd.Timestamp("2017-12-31")],
        "end_obs": [pd.Timestamp("2018-01-01 00:00:00")],
        "ref_epoch": [pd.Timestamp("2017-12-31")],
        "x": [constant.AE84], "y": [0.0], "z": [0.0],
        "vx": [0.0], "vy": [0.0], "vz": [0.0],
    })


def test_station_solution_uses_utc_limits_without_losing_tai_identity():
    frame = correct_receiver_epochs(pd.DataFrame({
        "station_code": ["TEST"], "obs_epoch": [pd.Timestamp("2018-01-01 00:00:10")],
        "receiver_clock_offset": [0.125],
    }))
    result = add_station_position(frame, _stations())
    assert result.sta_x.iloc[0] == constant.AE84
    assert result.obs_epoch.iloc[0] == pd.Timestamp("2018-01-01 00:00:10.125")
    assert result.attrs["obs_epoch_receiver_clock_corrected"] is True


def test_actual_two_stage_flow_selects_previous_utc_gim_and_keeps_ten_columns(tmp_path, monkeypatch):
    day = pd.Timestamp("2018-01-01")
    rows = 40
    rho = 1_336_000.0
    raw = pd.DataFrame({
        "obs_epoch": pd.date_range(day + pd.Timedelta(seconds=10), periods=rows, freq="3s"),
        "receiver_clock_offset": 0.125, "station_code": "TEST", "prn": "L39",
        "station_freq_shift": 0,
        "L1": (rho - constant.jason3_L1_pco - constant.starec_L1_pco) / constant.d_lambda1,
        "L2": (rho - constant.jason3_L2_pco - constant.starec_L2_pco) / constant.d_lambda2,
    })
    orbit = pd.DataFrame({
        "epoch": pd.date_range(day - pd.Timedelta(minutes=2), periods=8, freq="1min"),
        "x": constant.AE84 + rho, "y": 0.0, "z": 0.0, "prn": "L39",
    })
    stations = _stations()
    stations["end_obs"] = day + pd.Timedelta(days=1)
    monkeypatch.setattr(stage1, "load_block_aux", lambda *args, **kwargs: (orbit, stations))
    monkeypatch.setattr(stage1, "process_day", lambda *args, **kwargs: raw.copy())
    stec_root = tmp_path / "stec"
    stage1.process_block(day, day, sinex_file=tmp_path / "sinex", sp3_root=tmp_path / "sp3",
                         rinex_root=tmp_path / "rinex", output_root=stec_root, satellite="ja3")
    stec_path = stec_root / "2018/DOY001.pkl"
    stec = pd.read_pickle(stec_path)
    assert len(stec) == rows
    assert stec.attrs["pco_projection_model"] == "nominal_los_v1"
    assert stec.attrs["obs_epoch_receiver_clock_corrected"] is True
    assert stec.obs_epoch.iloc[0] == raw.obs_epoch.iloc[0] + pd.Timedelta(seconds=.125)

    selected_days = []
    def find(*, gim_dir, year, doy):
        selected_days.append((year, doy))
        return Path(gim_dir) / "reference.inx"
    monkeypatch.setattr(stage2, "find_covering_gim_file_from_dir", find)
    maps = GimMaps({pd.Timestamp("2017-12-31 22:00:00"): np.full((2, 2), 10.0),
                    pd.Timestamp("2018-01-01 00:00:00"): np.full((2, 2), 10.0)},
                   interval_seconds=7200.0)
    monkeypatch.setattr(stage2, "read_ionex_file", lambda _: (maps, [-90, 90], [-180, 180]))
    output_root = tmp_path / "vtec"
    stage2.process_day(day, stec_root, tmp_path / "gim", output_root, "ja3", DEFAULT_STAGE2_CONFIG)
    output = pd.read_pickle(output_root / "2018/DOY001.pickle")
    assert selected_days == [(2017, 365)]
    assert list(output) == ["obs_epoch", "ipp_lat", "ipp_lon", "stec", "vtec", "elevation",
                            "map_value", "station_code", "prn", "pass_id"]
    assert len(output) == rows
    np.testing.assert_allclose(output.vtec, 9.25, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(output.obs_epoch, stec.obs_epoch)
    assert output.attrs["obs_epoch_receiver_clock_corrected"] is True

    # A historical intermediate cannot silently combine old PCO STEC with
    # the repaired LOS detector background, or receive the clock twice.
    stec.attrs.clear()
    stec.to_pickle(stec_path)
    with pytest.raises(ValueError, match="Rebuild Stage1"):
        stage2.process_day(day, stec_root, tmp_path / "gim", output_root, "ja3", DEFAULT_STAGE2_CONFIG)


def _daily_vtec(root, day, *, corrected):
    path = root / str(day.year) / f"DOY{day.dayofyear:03d}.pickle"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"obs_epoch": [day + pd.Timedelta(hours=12)], "vtec": [10.0]})
    if corrected:
        frame.attrs.update(obs_epoch_time_scale="TAI", obs_epoch_receiver_clock_corrected=True)
    frame.to_pickle(path)


def test_historical_reader_retains_unknown_identity_but_strict_main_rejects_it(tmp_path):
    day = pd.Timestamp("2018-01-01")
    _daily_vtec(tmp_path, day, corrected=False)
    result = load_doris_cycle(tmp_path, day, day + pd.Timedelta(hours=23))
    assert result.attrs["obs_epoch_time_scale"] == "legacy_unspecified"
    with pytest.raises(ValueError, match="corrected-TAI evidence"):
        load_doris_cycle(tmp_path, day, day + pd.Timedelta(hours=23), require_corrected_tai=True)


def test_cycle_rejects_mixing_unknown_time_and_corrected_tai(tmp_path):
    day = pd.Timestamp("2018-01-01")
    _daily_vtec(tmp_path, day, corrected=True)
    _daily_vtec(tmp_path, day + pd.Timedelta(days=1), corrected=False)
    with pytest.raises(ValueError, match="Cannot mix"):
        load_doris_cycle(tmp_path, day, day + pd.Timedelta(days=2))
