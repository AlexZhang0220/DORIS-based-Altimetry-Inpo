"""Regression coverage for declared-interval GIM support checks."""

import numpy as np
import pandas as pd
import pytest

from doris_vtec.io.ionex_reader import GimMaps, read_ionex_file
from doris_vtec.stage2_stec2vtec.leveling.gim_interpolation import get_gim_vtec
import doris_inpo.io.gim_vtec_reader as cycle_reader


def _map(value):
    return np.full((2, 2), value, dtype=float)


def _frame(*epochs):
    return pd.DataFrame({
        "obs_epoch": pd.DatetimeIndex([pd.Timestamp(epoch) for epoch in epochs]),
        "ipp_lat": 0.0,
        "ipp_lon": 0.0,
    })


def _interpolate(maps, *epochs):
    return get_gim_vtec(
        maps, [-1.0, 1.0], [-1.0, 1.0], _frame(*epochs),
        observation_time_system="UTC",
    )


def test_missing_maps_do_not_bridge_eight_hour_gap():
    maps = GimMaps({
        pd.Timestamp("2018-01-01"): _map(10),
        pd.Timestamp("2018-01-01 08:00"): _map(30),
    }, interval_seconds=7200)
    result = _interpolate(maps, "2018-01-01 00:00", "2018-01-01 04:00", "2018-01-01 08:00")
    np.testing.assert_allclose(result, [10, np.nan, 30], equal_nan=True)


@pytest.mark.parametrize("interval_seconds", [900, 3600, 7200, 10800])
def test_declared_interval_is_used_without_new_fixed_threshold(interval_seconds):
    start = pd.Timestamp("2018-01-01")
    end = start + pd.Timedelta(seconds=interval_seconds)
    first = np.array([[0.0, 10.0], [20.0, 30.0]])
    maps = GimMaps({start: first, end: first + 20}, interval_seconds=interval_seconds)
    np.testing.assert_allclose(_interpolate(maps, start + (end - start) / 2), [25.0])


def test_gap_above_declared_interval_is_not_rounded_down():
    start = pd.Timestamp("2018-01-01")
    maps = GimMaps({
        start: _map(10),
        start + pd.Timedelta(seconds=7200, nanoseconds=1): _map(30),
    }, interval_seconds=7200)
    assert np.isnan(_interpolate(maps, start + pd.Timedelta(hours=1))[0])


def test_plain_dictionary_exact_epochs_and_outside_support_still_work():
    maps = {
        pd.Timestamp("2018-01-01"): _map(10),
        pd.Timestamp("2018-01-01 02:00"): _map(30),
    }
    result = _interpolate(maps, "2017-12-31 23:59", "2018-01-01", "2018-01-01 02:00", "2018-01-01 02:01")
    np.testing.assert_allclose(result, [np.nan, 10, 30, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="INTERVAL is required"):
        _interpolate(maps, "2018-01-01 01:00")


def test_exact_map_epoch_keeps_nanoseconds_without_temporal_blend():
    epoch = pd.Timestamp("2018-01-01 00:00:00.000000001")
    maps = {epoch: _map(12)}
    result = _interpolate(maps, epoch - pd.Timedelta(nanoseconds=1), epoch,
                          epoch + pd.Timedelta(nanoseconds=1))
    np.testing.assert_allclose(result, [np.nan, 12, np.nan], equal_nan=True)


@pytest.mark.parametrize("interval", [0, -1, np.nan, np.inf])
def test_invalid_metadata_rejected_only_when_temporal_blend_needed(interval):
    maps = GimMaps({
        pd.Timestamp("2018-01-01"): _map(10),
        pd.Timestamp("2018-01-01 02:00"): _map(30),
    }, interval_seconds=interval)
    np.testing.assert_array_equal(_interpolate(maps, "2018-01-01"), [10])
    with pytest.raises(ValueError, match="Invalid IONEX INTERVAL"):
        _interpolate(maps, "2018-01-01 01:00")


def _header_line(value, label):
    return f"{value:<60}{label}\n"


@pytest.mark.parametrize("interval", [None, "", "0", "-7200", "nan", "inf", "7200 3600", "invalid"])
def test_missing_or_invalid_header_interval_is_rejected(tmp_path, interval):
    path = tmp_path / "bad.inx"
    lines = [] if interval is None else [_header_line(interval, "INTERVAL")]
    path.write_text("".join(lines) + _header_line("", "END OF HEADER"))
    with pytest.raises(ValueError):
        read_ionex_file(path)


def test_duplicate_header_interval_is_rejected(tmp_path):
    path = tmp_path / "duplicate.inx"
    path.write_text(
        _header_line("7200", "INTERVAL") * 2
        + _header_line("", "END OF HEADER")
    )
    with pytest.raises(ValueError, match="Duplicate IONEX INTERVAL"):
        read_ionex_file(path)


def _mock_days(tmp_path, monkeypatch, maps_by_day):
    paths = {}
    for day in maps_by_day:
        stamp = pd.Timestamp(day)
        path = tmp_path / f"{stamp.year}-{stamp.dayofyear:03d}.inx"
        path.touch()
        paths[(stamp.year, stamp.dayofyear)] = path

    def read(path):
        day = next(day for day in maps_by_day if paths[(pd.Timestamp(day).year, pd.Timestamp(day).dayofyear)] == path)
        return maps_by_day[day], np.array([-1.0, 1.0]), np.array([-1.0, 1.0])

    monkeypatch.setattr(cycle_reader, "find_covering_gim_file_from_dir", lambda folder, year, doy: paths.get((year, doy)))
    monkeypatch.setattr(cycle_reader, "read_ionex_file", read)


def test_cycle_sorting_and_midnight_owner_preserve_interval(tmp_path, monkeypatch):
    midnight = pd.Timestamp("2018-01-02")
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": GimMaps({midnight: _map(2), pd.Timestamp("2018-01-01 22:00"): _map(1)}, interval_seconds=7200),
        "2018-01-02": GimMaps({pd.Timestamp("2018-01-02 02:00"): _map(20), midnight: _map(12)}, interval_seconds=7200),
    })
    maps, _, _ = cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00"), pd.Timestamp("2018-01-02 01:00"), observation_time_system="UTC")
    assert isinstance(maps, dict) and maps.interval_seconds == 7200
    assert list(maps) == sorted(maps)
    np.testing.assert_array_equal(maps[midnight], 12)
    np.testing.assert_allclose(_interpolate(maps, "2018-01-02 01:00"), [16])


def test_cycle_rejects_inconsistent_declared_intervals(tmp_path, monkeypatch):
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": GimMaps({pd.Timestamp("2018-01-01 22:00"): _map(1)}, interval_seconds=7200),
        "2018-01-02": GimMaps({pd.Timestamp("2018-01-02"): _map(2)}, interval_seconds=3600),
    })
    with pytest.raises(ValueError, match="Inconsistent IONEX INTERVAL"):
        cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00"), pd.Timestamp("2018-01-02"), observation_time_system="UTC")


@pytest.mark.parametrize("interval", [None, 0, -1, np.nan, np.inf])
def test_cycle_rejects_missing_or_invalid_interval_metadata(tmp_path, monkeypatch, interval):
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": GimMaps({pd.Timestamp("2018-01-01"): _map(1)}, interval_seconds=interval),
    })
    with pytest.raises(ValueError, match="IONEX INTERVAL metadata"):
        cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01"), observation_time_system="UTC")


def test_supplemented_midnight_file_cannot_lose_interval_metadata(tmp_path, monkeypatch):
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": GimMaps({pd.Timestamp("2018-01-01 22:00"): _map(1)}, interval_seconds=7200),
        "2018-01-02": {pd.Timestamp("2018-01-02"): _map(3)},
    })
    with pytest.raises(ValueError, match="Missing IONEX INTERVAL metadata"):
        cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00"), pd.Timestamp("2018-01-01 23:30"), observation_time_system="UTC")


def test_missing_midnight_map_leaves_large_neighbor_gap_as_nan(tmp_path, monkeypatch):
    _mock_days(tmp_path, monkeypatch, {
        "2018-01-01": GimMaps({pd.Timestamp("2018-01-01 22:00"): _map(1)}, interval_seconds=7200),
        "2018-01-02": GimMaps({pd.Timestamp("2018-01-02 02:00"): _map(3)}, interval_seconds=7200),
    })
    maps, _, _ = cycle_reader.load_gim_cycle(tmp_path, pd.Timestamp("2018-01-01 23:00"), pd.Timestamp("2018-01-02 01:00"), observation_time_system="UTC")
    np.testing.assert_allclose(_interpolate(maps, "2018-01-01 23:00", "2018-01-02 01:00"), [np.nan, np.nan], equal_nan=True)
