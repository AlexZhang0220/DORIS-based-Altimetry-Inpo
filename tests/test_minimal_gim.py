import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from doris_vtec.io.ionex_reader import GimMaps, read_ionex_file
from doris_vtec.io.utils import tai_epochs_to_utc
from doris_vtec.stage2_stec2vtec.leveling.gim_interpolation import get_gim_vtec
import doris_inpo.io.gim_vtec_reader as cycle_reader


def _line(content, label):
    return f"{content:<60}{label}\n"


def _ionex(tmp_path, *, count=13, lat_step=-2.5, exponent=-1, missing=9999):
    lines = [
        _line(str(count), "# OF MAPS IN FILE"),
        _line(f"87.5 -87.5 {lat_step}", "LAT1 / LAT2 / DLAT"),
        _line("-180 180 5", "LON1 / LON2 / DLON"),
        _line("7200", "INTERVAL"),
        _line(str(exponent), "EXPONENT"),
        _line("", "END OF HEADER"),
    ]
    for index in range(13):
        epoch = dt.datetime(2018, 1, 1) + dt.timedelta(hours=2 * index)
        lines.extend([
            _line(str(index + 1), "START OF TEC MAP"),
            _line(epoch.strftime("%Y %m %d %H %M %S"), "EPOCH OF CURRENT MAP"),
        ])
        for latitude in np.linspace(87.5, -87.5, 71):
            row = "  " + "".join(f"{value:6.1f}" for value in (latitude, -180, 180, 5, 450))
            lines.append(_line(row, "LAT/LON1/LON2/DLON/H"))
            values = [100] * 73
            values[0] = missing
            for offset in range(0, 73, 16):
                lines.append("".join(f"{value:5d}" for value in values[offset:offset + 16]) + "\n")
        lines.append(_line(str(index + 1), "END OF TEC MAP"))
    path = tmp_path / "igsg0010.18i"
    path.write_text("".join(lines))
    return path


def test_missing_code_removed_before_exponent_and_tuple_preserved(tmp_path):
    gim, lat, lon = read_ionex_file(_ionex(tmp_path))
    assert len(gim) == 13 and lat.shape == (71,) and lon.shape == (73,)
    assert isinstance(gim, dict) and gim.interval_seconds == 7200
    first = gim[dt.datetime(2018, 1, 1)]
    assert np.isnan(first[:, 0]).all()
    np.testing.assert_array_equal(first[:, 1:], 10.0)


@pytest.mark.parametrize("kwargs", [{"count": 25}, {"lat_step": -5.0}])
def test_unsupported_header_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError, match="Unsupported IONEX"):
        read_ionex_file(_ionex(tmp_path, **kwargs))


def test_actual_map_count_rejected(tmp_path):
    path = _ionex(tmp_path)
    text = path.read_text()
    path.write_text(text.replace("START OF TEC MAP", "START OF RMS MAP", 1))
    with pytest.raises(ValueError, match="exactly 13 TEC maps"):
        read_ionex_file(path)


def test_actual_row_grid_rejected(tmp_path):
    path = _ionex(tmp_path)
    row = "  " + "".join(f"{value:6.1f}" for value in (87.5, -180, 180, 5, 450))
    bad = "  " + "".join(f"{value:6.1f}" for value in (85, -180, 180, 5, 450))
    path.write_text(path.read_text().replace(row, bad, 1))
    with pytest.raises(ValueError, match="inconsistent IONEX map row"):
        read_ionex_file(path)


def _frame(epochs):
    return pd.DataFrame({"obs_epoch": pd.to_datetime(epochs), "ipp_lat": 0.0, "ipp_lon": 0.0})


def _maps(start="2018-01-01 00:00:00", end="2018-01-01 02:00:00"):
    return GimMaps({pd.Timestamp(start): np.full((2, 2), 10.0),
                    pd.Timestamp(end): np.full((2, 2), 30.0)}, interval_seconds=7200)


def test_tai_is_converted_before_interpolation_and_no_temporal_extrapolation():
    frame = _frame(["2018-01-01 00:00:36", "2018-01-01 00:00:37",
                    "2018-01-01 01:00:37", "2018-01-01 02:00:37", "2018-01-01 02:00:38"])
    result = get_gim_vtec(_maps(), [-1, 1], [-1, 1], frame)
    np.testing.assert_allclose(result, [np.nan, 10, 20, 30, np.nan], equal_nan=True)


def test_explicit_utc_and_spatial_bilinear_interpolation():
    gim = {pd.Timestamp("2018-01-01"): np.array([[0.0, 10.0], [20.0, 30.0]])}
    result = get_gim_vtec(gim, [-1, 1], [-1, 1], _frame(["2018-01-01"]), observation_time_system="UTC")
    np.testing.assert_allclose(result, [15.0])
    np.testing.assert_array_equal(get_gim_vtec({}, [-1, 1], [-1, 1], _frame(["2018-01-01"])), [np.nan])
    with pytest.raises(ValueError, match="observation_time_system"):
        get_gim_vtec(gim, [-1, 1], [-1, 1], _frame(["2018-01-01"]), observation_time_system="GPS")


def test_historical_leap_second_boundary_is_continuous_for_file_labels():
    result = np.asarray(tai_epochs_to_utc(pd.to_datetime([
        "2017-01-01 00:00:35", "2017-01-01 00:00:36", "2017-01-01 00:00:37",
    ])), dtype="datetime64[ns]")
    assert result[0] == np.datetime64("2016-12-31T23:59:59", "ns")
    assert result[1].astype("datetime64[D]") == np.datetime64("2016-12-31")
    assert result[2] == np.datetime64("2017-01-01T00:00:00", "ns")


def test_last_nanosecond_of_leap_second_keeps_previous_utc_day():
    result = tai_epochs_to_utc(pd.to_datetime(["2017-01-01 00:00:36.999999999"]))
    assert result[0] == np.datetime64("2016-12-31T23:59:59.999999999", "ns")


def _mock_days(tmp_path, monkeypatch, maps_by_day, grids_by_day=None):
    loaded = []
    paths = {}
    for day in maps_by_day:
        stamp = pd.Timestamp(day)
        path = tmp_path / f"{stamp.year}-{stamp.dayofyear:03d}.inx"
        path.touch()
        paths[(stamp.year, stamp.dayofyear)] = path
    def find(folder, year, doy):
        loaded.append((year, doy))
        return paths.get((year, doy))
    def read(path):
        stamp = next(day for day in maps_by_day if paths[(pd.Timestamp(day).year, pd.Timestamp(day).dayofyear)] == path)
        lat, lon = ([-1.0, 1.0], [-1.0, 1.0])
        if grids_by_day and stamp in grids_by_day:
            lat, lon = grids_by_day[stamp]
        maps = maps_by_day[stamp]
        if not isinstance(maps, GimMaps):
            maps = GimMaps(maps, interval_seconds=7200)
        return maps, np.array(lat), np.array(lon)
    monkeypatch.setattr(cycle_reader, "find_covering_gim_file_from_dir", find)
    monkeypatch.setattr(cycle_reader, "read_ionex_file", read)
    return loaded


def _map(value):
    return np.full((2, 2), value, dtype=float)


def test_tai_boundaries_select_previous_utc_year_and_midnight_owner(tmp_path, monkeypatch):
    midnight = pd.Timestamp("2017-01-01")
    loaded = _mock_days(tmp_path, monkeypatch, {
        "2016-12-31": {pd.Timestamp("2016-12-31 22:00:00"): _map(1), midnight: _map(2)},
        "2017-01-01": {midnight: _map(12), pd.Timestamp("2017-01-01 02:00:00"): _map(13)},
    })
    gim, lat, lon = cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2017-01-01 00:00:35"), pd.Timestamp("2017-01-01 00:01:37"))
    assert loaded == [(2016, 366), (2017, 1)]
    np.testing.assert_array_equal(gim[midnight], 12)
    assert lat.shape == lon.shape == (2,)


def test_neighbor_loaded_only_when_needed_for_bracketing(tmp_path, monkeypatch):
    loaded = _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": {pd.Timestamp("2018-01-01 22:00:00"): _map(1)},
        "2018-01-02": {pd.Timestamp("2018-01-02 00:00:00"): _map(3)},
    })
    gim, lat, lon = cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00:00"), pd.Timestamp("2018-01-01 23:30:00"), observation_time_system="UTC")
    assert loaded == [(2018, 1), (2018, 2)]
    result = get_gim_vtec(gim, lat, lon, _frame(["2018-01-01 23:00:00"]), observation_time_system="UTC")
    np.testing.assert_allclose(result, [2.0])


def test_non_midnight_conflict_rejected(tmp_path, monkeypatch):
    collision = pd.Timestamp("2018-01-02 01:00:00")
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": {collision: _map(1)},
        "2018-01-02": {collision: _map(2)},
    })
    with pytest.raises(ValueError, match="Conflicting IONEX maps"):
        cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00"), pd.Timestamp("2018-01-02 01:00"), observation_time_system="UTC")


def test_cross_file_grid_mismatch_rejected(tmp_path, monkeypatch):
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": {pd.Timestamp("2018-01-01 22:00"): _map(1)},
        "2018-01-02": {pd.Timestamp("2018-01-02 02:00"): _map(2)},
    }, {"2018-01-02": ([-2.0, 2.0], [-1.0, 1.0])})
    with pytest.raises(ValueError, match="Inconsistent IONEX grids"):
        cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00"), pd.Timestamp("2018-01-02 01:00"), observation_time_system="UTC")


def test_missing_cycle_files_fail_clearly_without_none_exists_error(tmp_path, monkeypatch):
    _mock_days(tmp_path, monkeypatch, {})
    with pytest.raises(RuntimeError, match="No GIM files"):
        cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01"), observation_time_system="UTC")
