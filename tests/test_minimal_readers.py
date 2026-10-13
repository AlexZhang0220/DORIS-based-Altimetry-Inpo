"""Regressions for bounded reader fixes; existing science formulas stay fixed."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import constant
from doris_vtec.io import rinex_reader, sinex_reader


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]


def header_line(content, label):
    return f"{content:<60}{label:<20}\n"


def rinex_header(scale_records=()):
    station = list(" " * 60)
    station[1:3] = "01"
    station[5:9] = "TEST"
    station[52:56] = " -15"
    return rinex_reader.parse_rinex_header([
        header_line("JASON-3", "SATELLITE NAME"),
        header_line("D    4  L1  L2  C1  C2", "SYS / # / OBS TYPES"),
        *(header_line(record, "SYS / SCALE FACTOR") for record in scale_records),
        header_line("".join(station), "STATION REFERENCE"),
        header_line("", "END OF HEADER"),
    ])


def epoch_record(clock):
    line = list(" " * 58)
    line[0] = ">"
    line[2:31] = "2024 05 01 00 00 00.000000000"
    line[32:35] = "  0"
    line[35:38] = "  1"
    line[43:56] = f"{clock:13.9f}"
    line[57] = "0"
    return "".join(line) + "\n"


def observation_record(values=(123.0, 456.0, 3.0, 4.0)):
    return "D01" + "".join(f"{value:14.3f}  " for value in values) + "\n"


def parse_observation(header, clock=0.0):
    return rinex_reader.parse_rinex_obs(
        [epoch_record(clock), observation_record()], header
    ).iloc[0]


def test_imports_are_from_independent_candidate():
    for module in (constant, rinex_reader, sinex_reader):
        assert Path(module.__file__).resolve().is_relative_to(CANDIDATE_ROOT)


@pytest.mark.parametrize("clock", [-12.345678901, 123.456789012, 0.0])
def test_complete_receiver_clock_field_preserves_sign_and_leading_digit(clock):
    row = parse_observation(rinex_header(), clock)
    assert row.receiver_clock_offset == pytest.approx(clock, abs=1e-12)
    # The root workflow applies clock correction; this reader keeps its old API.
    assert row.obs_epoch == pd.Timestamp("2024-05-01")


def test_overflowed_receiver_clock_field_is_rejected_instead_of_truncated():
    line = epoch_record(0.0)
    overflow = line[:43] + "1234.567890123" + "4 0\n"
    with pytest.raises(ValueError, match="F13.9"):
        rinex_reader.parse_rinex_obs(
            [overflow, observation_record()], rinex_header()
        )


@pytest.mark.parametrize("factor", [1, 10, 100, 1000])
def test_pseudorange_scaling_uses_declared_factor_and_returns_meters(factor):
    row = parse_observation(rinex_header([f"D {factor} 2 C1 C2"]))
    assert row.C1 == pytest.approx(3.0 / factor * 1000.0)
    assert row.C2 == pytest.approx(4.0 / factor * 1000.0)
    assert row.L1 == 123.0
    assert row.L2 == 456.0


def test_factor_100_keeps_historical_standard_pseudorange_values():
    row = parse_observation(rinex_header(["D 100 2 C1 C2"]))
    assert row.C1 == 3.0 * 10.0
    assert row.C2 == 4.0 * 10.0


def test_phase_scaling_restores_cycles_without_changing_pseudorange_scale():
    row = parse_observation(rinex_header([
        "D 10 2 L1 L2", "D 100 2 C1 C2"
    ]))
    assert row.L1 == pytest.approx(12.3)
    assert row.L2 == pytest.approx(45.6)
    assert row.C1 == 30.0
    assert row.C2 == 40.0


def test_zero_observation_count_applies_scale_to_all_declared_observables():
    row = parse_observation(rinex_header(["D 10 0"]))
    assert row.L1 == pytest.approx(12.3)
    assert row.L2 == pytest.approx(45.6)
    assert row.C1 == 300.0
    assert row.C2 == 400.0


def test_missing_scale_declaration_means_standard_unit_factor():
    row = parse_observation(rinex_header())
    assert row.C1 == 3000.0
    assert row.C2 == 4000.0
    assert row.station_freq_shift == -15


@pytest.mark.parametrize("factor", ["0", "-1", "nan", "inf"])
def test_invalid_declared_scale_is_rejected(factor):
    with pytest.raises(ValueError, match="scale factor"):
        rinex_header([f"D {factor} 2 C1 C2"])


@pytest.mark.parametrize("record, expected_start, expected_end", [
    (
        " ADEA  A    1 D 93:003:00000 98:084:11545 95:226:48972",
        "1993-01-03T00:00:00", "1998-03-25T03:12:25",
    ),
    (
        " ARLA  A    1 D 87:339:00000 92:343:86399 90:157:00000",
        "1987-12-05T00:00:00", "1992-12-08T23:59:59",
    ),
])
def test_existing_sinex_validity_slices_read_full_native_records(
    record, expected_start, expected_end
):
    # Real DPOD2020/DPOD2014 layout: the two slice choices differ only in padding.
    assert record[15:28].strip() == record[16:29].strip()
    assert record[28:41].strip() == record[29:42].strip()
    row = sinex_reader.parse_solution_epochs(["*CODE", record]).iloc[0]
    assert row.start_obs == pd.Timestamp(expected_start)
    assert row.end_obs == pd.Timestamp(expected_end)


def test_nominal_frequencies_wavelengths_and_iono_factors_remain_unchanged():
    assert constant.d_f1 == 2_036_250_000.0
    assert constant.d_f2 == 401_250_000.0
    assert constant.d_lambda1 == pytest.approx(0.14722772645794966)
    assert constant.d_lambda2 == pytest.approx(0.7471463127725857)
    assert constant.d_k == pytest.approx(0.04039872971065632)
    assert constant.d_w == constant.d_f1**2 / 40.3


@pytest.mark.parametrize("relative_path, expected_count", [
    ("doris_vtec/stage1_obs2stec/ionosphere/dion.py", 6),
    ("doris_vtec/stage2_stec2vtec/pass_detection/doris_cycle_slip.py", 4),
])
def test_station_frequency_formulas_keep_original_values(relative_path, expected_count):
    shifts = np.array([-15.0, 0.0, 18.0])
    namespace = {"constant": constant, "c": constant.c, "shift": shifts}
    selected = []
    for node in ast.walk(ast.parse((CANDIDATE_ROOT / relative_path).read_text())):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {
                "d_f1", "d_f2", "d_lambda1", "d_lambda2", "d_k", "d_w"
            }
        ):
            selected.append(node)
    assert len(selected) == expected_count
    for node in sorted(selected, key=lambda node: node.lineno):
        exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), namespace)
    scale = 1.0 + shifts * (4.0 / 3.0 * (87.0 / 5.0 / 2**26))
    np.testing.assert_allclose(namespace["d_f1"], 2_036_250_000.0 * scale, rtol=1e-15)
    np.testing.assert_allclose(namespace["d_f2"], 401_250_000.0 * scale, rtol=1e-15)
    np.testing.assert_allclose(namespace["d_lambda1"], constant.c / namespace["d_f1"], rtol=1e-15)
    np.testing.assert_allclose(namespace["d_lambda2"], constant.c / namespace["d_f2"], rtol=1e-15)
    assert namespace["d_f1"][0] == pytest.approx(2_036_239_440.8136606)
    assert namespace["d_f2"][0] == pytest.approx(401_247_919.2763567)
    if "d_w" in namespace:
        np.testing.assert_allclose(namespace["d_k"], constant.d_k, rtol=1e-15)
        np.testing.assert_allclose(namespace["d_w"], namespace["d_f1"]**2 / 40.3, rtol=1e-15)


def test_starec_and_alcatel_combined_pco_constants_do_not_overwrite_each_other():
    gamma = (2_036_250_000.0 / 401_250_000.0)**2
    starec = (gamma * (0.315 + 0.487) - (0.147 + 0.000)) / (gamma - 1.0)
    alcatel = (gamma * (0.315 + 0.510) - (0.147 + 0.335)) / (gamma - 1.0)
    assert constant.starec_ja3_iono_comb_pco == pytest.approx(starec)
    assert constant.alcatel_ja3_iono_comb_pco == pytest.approx(alcatel)
    assert starec != pytest.approx(alcatel)
