"""Cycle-edge support and exact-location IDW without changing scientific windows."""

import numpy as np
import pandas as pd
import pytest

from doris_inpo.config import DEFAULT_INPO_CONFIG
from doris_inpo.io.doris_vtec_reader import load_doris_cycle
from doris_inpo.interpolation.doris_based_alt_inpo import filter_doris_in_time
from doris_inpo.interpolation.weights import idw
from doris_inpo.main import process_single_pass


def _write_daily(root, frame, *, corrected=True):
    for day, rows in frame.groupby(frame["obs_epoch"].dt.normalize()):
        path = root / str(day.year) / f"DOY{day.dayofyear:03d}.pickle"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = rows.copy()
        if corrected:
            rows.attrs.update(obs_epoch_time_scale="TAI", obs_epoch_receiver_clock_corrected=True)
        rows.to_pickle(path)


def test_halo_reads_both_neighboring_days_across_year_and_preserves_default(tmp_path):
    start = pd.Timestamp("2018-12-31")
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    epochs = [start - pd.Timedelta(seconds=301), start - pd.Timedelta(seconds=300),
              start, end, end + pd.Timedelta(seconds=300), end + pd.Timedelta(seconds=301)]
    frame = pd.DataFrame({"obs_epoch": epochs, "vtec": np.arange(6, dtype=float)})
    _write_daily(tmp_path, frame)

    old_selection = load_doris_cycle(tmp_path, start, end, require_corrected_tai=True)
    padded = load_doris_cycle(tmp_path, start, end, require_corrected_tai=True,
                              time_margin_sec=300.0)

    assert old_selection["obs_epoch"].tolist() == epochs[2:4]
    assert padded["obs_epoch"].tolist() == epochs[1:5]
    assert padded.attrs["obs_epoch_time_scale"] == "TAI"
    assert padded.attrs["obs_epoch_receiver_clock_corrected"] is True


@pytest.mark.parametrize("edge", ["start", "end"])
def test_halo_restores_thirty_point_support_without_expanding_altimetry_targets(tmp_path, edge):
    start = pd.Timestamp("2018-01-01")
    end = start + pd.Timedelta(hours=1)
    target = start if edge == "start" else end
    epochs = target + pd.to_timedelta(np.arange(-145, 146, 10), unit="s")
    doris = pd.DataFrame({"obs_epoch": epochs, "ipp_lat": 1.0, "ipp_lon": 0.0,
                          "vtec": 20.0, "elevation": 30.0})
    _write_daily(tmp_path, doris)
    unpadded = load_doris_cycle(tmp_path, start, end, require_corrected_tai=True)
    padded = load_doris_cycle(tmp_path, start, end, require_corrected_tai=True,
                              time_margin_sec=300.0)
    alt = pd.DataFrame({"obs_epoch": [target], "ipp_lat": [0.0], "ipp_lon": [0.0],
                        "pass_in_cycle": [1]})

    before = process_single_pass(1, alt, unpadded, DEFAULT_INPO_CONFIG, 70)
    after = process_single_pass(1, alt, padded, DEFAULT_INPO_CONFIG, 70)

    assert len(unpadded) == 15
    assert len(padded) == 30
    assert np.isnan(before["doris_vtec_idw"].iloc[0])
    assert after["doris_vtec_idw"].iloc[0] == pytest.approx(20.0)
    assert after["doris_points_count"].iloc[0] == 30
    pd.testing.assert_frame_equal(after[list(alt)], alt)


def test_pass_prefilter_still_excludes_data_outside_five_minutes(monkeypatch):
    import doris_inpo.main as pipeline

    epoch = pd.Timestamp("2018-01-01")
    offsets = [-301, -300, 0, 300, 301]
    doris = pd.DataFrame({"obs_epoch": epoch + pd.to_timedelta(offsets, unit="s")})
    alt = pd.DataFrame({"obs_epoch": [epoch], "pass_in_cycle": [1]})
    captured = []

    def interpolate(targets, candidates, settings):
        captured.extend(candidates["obs_epoch"].tolist())
        return targets.copy()

    monkeypatch.setattr(pipeline, "altimetry_inpo", interpolate)
    process_single_pass(1, alt, doris, DEFAULT_INPO_CONFIG, 70)
    assert captured == doris["obs_epoch"].iloc[1:4].tolist()


def test_point_filter_retains_original_strict_four_hundred_second_limit():
    epoch = pd.Timestamp("2018-01-01")
    frame = pd.DataFrame({"obs_epoch": epoch + pd.to_timedelta([-400, -399, 0, 399, 400], unit="s")})
    assert filter_doris_in_time(frame, epoch)["obs_epoch"].tolist() == frame["obs_epoch"].iloc[1:4].tolist()


def test_halo_does_not_relabel_historical_time_identity(tmp_path):
    epoch = pd.Timestamp("2018-01-01")
    frame = pd.DataFrame({"obs_epoch": [epoch - pd.Timedelta(seconds=1)], "vtec": [10.0]})
    _write_daily(tmp_path, frame, corrected=False)
    result = load_doris_cycle(tmp_path, epoch, epoch + pd.Timedelta(hours=1), time_margin_sec=300.0)
    assert result.attrs["obs_epoch_time_scale"] == "legacy_unspecified"
    assert "obs_epoch_receiver_clock_corrected" not in result.attrs
    with pytest.raises(ValueError, match="corrected-TAI evidence"):
        load_doris_cycle(tmp_path, epoch, epoch + pd.Timedelta(hours=1),
                         time_margin_sec=300.0, require_corrected_tai=True)


@pytest.mark.parametrize("margin", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_halo_is_rejected(tmp_path, margin):
    epoch = pd.Timestamp("2018-01-01")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        load_doris_cycle(tmp_path, epoch, epoch, time_margin_sec=margin)


@pytest.mark.parametrize("distances, values, expected", [
    ([0, 10], [20, 40], 20.0),
    ([0, 0], [20, 30], 25.0),
    ([0, 0, 10], [np.nan, 20, 40], 20.0),
    ([0, 10], [np.nan, 40], 40.0),
    ([1, 2, 4], [10, 20, 40], 120.0 / 7.0),
    ([1, 2, np.nan, np.inf, -1], [10, 20, 100, 100, 100], 40.0 / 3.0),
])
def test_idw_exact_locations_and_original_inverse_distance_weights(distances, values, expected):
    assert idw(distances, values) == pytest.approx(expected)


@pytest.mark.parametrize("distances, values", [([], []), ([0], [np.nan]), ([np.nan], [20])])
def test_idw_without_valid_support_remains_nan(distances, values):
    assert np.isnan(idw(distances, values))
