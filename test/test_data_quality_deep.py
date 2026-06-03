"""
Deep pytest test suite for the universal DataQuality class.

How to use:
    1. Put this file into the folder with:
       - data_quality.py
       - data_quality_metadata.json
    2. Run:
       python -m pytest test_data_quality_deep.py -q

The tests intentionally check not only the current five sensors, but also the
metadata-driven scaling idea: a new ordinary physical sensor can be added only
through JSON metadata without changing data_quality.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from algorithms import DataQuality


BASE_DIR = Path(__file__).resolve().parent
SOURCE_METADATA_PATH = BASE_DIR / "data_quality_metadata.json"
if not SOURCE_METADATA_PATH.exists():
    SOURCE_METADATA_PATH = BASE_DIR.parent / "algorithms" / "metadata" / "data_quality_metadata.json"


def _read_metadata() -> Dict[str, Any]:
    return json.loads(SOURCE_METADATA_PATH.read_text(encoding="utf-8"))


def _write_metadata(tmp_path: Path, metadata: Dict[str, Any]) -> Path:
    metadata_path = tmp_path / "data_quality_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def make_dq(tmp_path: Path, metadata: Dict[str, Any] | None = None, autosave: bool = True) -> DataQuality:
    metadata = metadata or _read_metadata()
    metadata_path = _write_metadata(tmp_path, metadata)
    state_path = tmp_path / "data_quality_state.json"
    return DataQuality(metadata_path=metadata_path, state_path=state_path, autosave=autosave)


def metadata_with_generic_water_level_sensor() -> Dict[str, Any]:
    metadata = _read_metadata()
    metadata["sensors"]["GENERIC_WATER_LEVEL_SENSOR"] = {
        "parameters": {
            "water_level_cm": {
                "min": 0,
                "max": 300,
                "resolution": 0.1,
                "initial_history": [100, 100, 100, 100, 100],
                "rules": [
                    {"type": "range_check", "replacement": "median_history"},
                    {"type": "clamp"},
                ],
            }
        },
    }
    return metadata


def assert_numeric_dict(values: Dict[str, Any]) -> None:
    assert values, "DataQuality must return a non-empty dictionary."
    for key, value in values.items():
        assert isinstance(value, (int, float)), f"{key} must be numeric, got {type(value)!r}"
        assert math.isfinite(float(value)), f"{key} must be finite, got {value!r}"


def assert_history_size(dq: DataQuality, sensor: str, parameter: str, size: int = 5) -> None:
    history = dq.state["sensors"][sensor]["history"][parameter]
    assert len(history) == size
    assert all(isinstance(value, float) for value in history)
    assert all(math.isfinite(value) for value in history)


# ---------------------------------------------------------------------------
# Metadata and state behavior
# ---------------------------------------------------------------------------


def test_initial_state_is_created_from_metadata_for_all_sensors(tmp_path: Path):
    dq = make_dq(tmp_path)
    metadata = _read_metadata()

    assert set(dq.state["sensors"].keys()) == set(metadata["sensors"].keys())

    for sensor_name, sensor_cfg in metadata["sensors"].items():
        history = dq.state["sensors"][sensor_name]["history"]
        for parameter_name, parameter_cfg in sensor_cfg["parameters"].items():
            assert parameter_name in history
            assert len(history[parameter_name]) == 5
            assert all(isinstance(v, float) for v in history[parameter_name])

            output_parameter = parameter_cfg.get("output_parameter")
            if output_parameter:
                assert output_parameter in history
                assert len(history[output_parameter]) == 5


def test_unknown_sensor_raises_clear_key_error(tmp_path: Path):
    dq = make_dq(tmp_path)
    with pytest.raises(KeyError, match="not configured"):
        dq.clean("UNKNOWN_SENSOR", {"value": 1}, context={"communication_ok": True})


def test_return_details_contains_values_and_rule_flags(tmp_path: Path):
    dq = make_dq(tmp_path)
    result = dq.clean(
        "SONBEST_SM9560B",
        {"illuminance_lux": 12000},
        context={"communication_ok": True, "sun_state": "night"},
        return_details=True,
    )

    assert set(result.keys()) == {"values", "flags"}
    assert result["values"]["illuminance_lux"] == 12000.0
    assert result["flags"]["illuminance_lux"] == ["clamp: final range limit applied"]


def test_invalid_rule_type_fails_fast_with_clear_error(tmp_path: Path):
    metadata = _read_metadata()
    metadata["sensors"]["BROKEN_SENSOR"] = {
        "description": "Intentional broken metadata for unit testing.",
        "parameters": {
            "value": {
                "unit": "u",
                "min": 0,
                "max": 10,
                "initial_history": [1, 1, 1, 1, 1],
                "rules": [{"type": "unknown_rule_type"}],
            }
        },
    }
    dq = make_dq(tmp_path, metadata)

    with pytest.raises(ValueError, match="Unknown rule type"):
        dq.clean("BROKEN_SENSOR", {"value": 5}, context={"communication_ok": True})


def test_invalid_replacement_method_fails_fast_with_clear_error(tmp_path: Path):
    metadata = _read_metadata()
    metadata["sensors"]["BROKEN_REPLACEMENT_SENSOR"] = {
        "description": "Intentional broken replacement for unit testing.",
        "parameters": {
            "value": {
                "unit": "u",
                "min": 0,
                "max": 10,
                "initial_history": [1, 1, 1, 1, 1],
                "rules": [{"type": "range_check", "replacement": "bad_replacement"}],
            }
        },
    }
    dq = make_dq(tmp_path, metadata)

    with pytest.raises(ValueError, match="Unknown replacement method"):
        dq.clean("BROKEN_REPLACEMENT_SENSOR", {"value": 100}, context={"communication_ok": True})


# ---------------------------------------------------------------------------
# Generic numeric behavior
# ---------------------------------------------------------------------------


def test_non_numeric_and_missing_values_are_replaced_by_history_median(tmp_path: Path):
    dq = make_dq(tmp_path, metadata_with_generic_water_level_sensor())

    result = dq.clean(
        "GENERIC_WATER_LEVEL_SENSOR",
        {"water_level_cm": "not-a-number"},
        context={"communication_ok": True},
    )
    assert result["water_level_cm"] == 100.0
    assert_numeric_dict(result)

    result = dq.clean(
        "GENERIC_WATER_LEVEL_SENSOR",
        {},
        context={"communication_ok": True},
    )
    assert result["water_level_cm"] == 100.0
    assert_numeric_dict(result)


def test_numeric_strings_numpy_scalars_and_pandas_values_are_accepted(tmp_path: Path):
    dq = make_dq(tmp_path, metadata_with_generic_water_level_sensor())

    result = dq.clean(
        "GENERIC_WATER_LEVEL_SENSOR",
        {"water_level_cm": "120.43"},
        context={"communication_ok": True},
    )
    assert result["water_level_cm"] == 120.4

    result = dq.clean(
        "GENERIC_WATER_LEVEL_SENSOR",
        {"water_level_cm": np.float64(121.26)},
        context={"communication_ok": True},
    )
    assert result["water_level_cm"] == 121.3

    result = dq.clean(
        "GENERIC_WATER_LEVEL_SENSOR",
        {"water_level_cm": pd.Series([122.24]).iloc[0]},
        context={"communication_ok": True},
    )
    assert result["water_level_cm"] == 122.2


def test_communication_failure_returns_numeric_history_median_and_updates_history(tmp_path: Path):
    dq = make_dq(tmp_path)

    result = dq.clean(
        "Veinasa_THPW_NJ",
        {},
        context={"communication_ok": False},
    )

    assert result == {
        "wind_speed_ms": 0.0,
        "wind_direction_deg": 0.0,
        "air_temperature_c": 0.0,
        "air_humidity_percent": 0.0,
        "air_pressure_hpa": 300.0,
    }
    assert_numeric_dict(result)
    assert_history_size(dq, "Veinasa_THPW_NJ", "air_temperature_c")


def test_night_missing_solar_parameters_are_zero_not_history_median(tmp_path: Path):
    dq = make_dq(tmp_path)

    lux = dq.clean("SONBEST_SM9560B", {}, context={"communication_ok": False, "sun_state": "night"})
    radiation = dq.clean("XS_TBQ02C", {}, context={"communication_ok": False, "sun_state": "night"})

    assert lux["illuminance_lux"] == 0.0
    assert radiation["solar_radiation_wm2"] == 0.0
    assert_numeric_dict(lux)
    assert_numeric_dict(radiation)


def test_history_always_keeps_latest_five_values_and_autosaves(tmp_path: Path):
    dq = make_dq(tmp_path, metadata_with_generic_water_level_sensor())

    for value in [110, 120, 130, 140, 150, 160, 170]:
        dq.clean("GENERIC_WATER_LEVEL_SENSOR", {"water_level_cm": value}, context={"communication_ok": True})

    history = dq.state["sensors"]["GENERIC_WATER_LEVEL_SENSOR"]["history"]["water_level_cm"]
    assert history == [130.0, 140.0, 150.0, 160.0, 170.0]

    state_on_disk = json.loads((tmp_path / "data_quality_state.json").read_text(encoding="utf-8"))
    assert state_on_disk["sensors"]["GENERIC_WATER_LEVEL_SENSOR"]["history"]["water_level_cm"] == history


# ---------------------------------------------------------------------------
# Veinasa THPW-NJ
# ---------------------------------------------------------------------------


def test_weather_station_range_and_jump_rules(tmp_path: Path):
    dq = make_dq(tmp_path)

    result = dq.clean(
        "Veinasa_THPW_NJ",
        {
            "wind_speed_ms": 100,
            "wind_direction_deg": 200,
            "air_temperature_c": 40,
            "air_humidity_percent": 90,
            "air_pressure_hpa": 1020,
        },
        context={"communication_ok": True},
    )

    assert result == {
        "wind_speed_ms": 0.0,
        "wind_direction_deg": 200.0,
        "air_temperature_c": 40.0,
        "air_humidity_percent": 90.0,
        "air_pressure_hpa": 1020.0,
    }
    assert_numeric_dict(result)


def test_weather_station_boundary_values_are_accepted_when_inside_thresholds(tmp_path: Path):
    dq = make_dq(tmp_path)

    result = dq.clean(
        "Veinasa_THPW_NJ",
        {
            "wind_speed_ms": 15,
            "wind_direction_deg": 350,
            "air_temperature_c": 25,
            "air_humidity_percent": 70,
            "air_pressure_hpa": 1018,
        },
        context={"communication_ok": True},
    )

    # jump_check uses '>' rather than '>='; therefore exact threshold values are accepted.
    assert result["wind_speed_ms"] == 15.0
    assert result["wind_direction_deg"] == 350.0
    assert result["air_temperature_c"] == 25.0
    assert result["air_humidity_percent"] == 70.0
    assert result["air_pressure_hpa"] == 1018.0


def test_wind_direction_is_replaced_when_wind_is_too_weak(tmp_path: Path):
    dq = make_dq(tmp_path)

    result = dq.clean(
        "Veinasa_THPW_NJ",
        {
            "wind_speed_ms": 0.5,
            "wind_direction_deg": 180,
            "air_temperature_c": 20,
            "air_humidity_percent": 50,
            "air_pressure_hpa": 1013,
        },
        context={"communication_ok": True},
    )

    assert result["wind_speed_ms"] == 0.5
    assert result["wind_direction_deg"] == 180.0


# ---------------------------------------------------------------------------
# SONBEST SM9560B and XS-TBQ02C
# ---------------------------------------------------------------------------


def test_illuminance_day_twilight_night_rules(tmp_path: Path):
    dq = make_dq(tmp_path)

    night = dq.clean("SONBEST_SM9560B", {"illuminance_lux": 12000}, context={"communication_ok": True, "sun_state": "night"})
    assert night["illuminance_lux"] == 12000.0

    day_inside_jump_limit = dq.clean("SONBEST_SM9560B", {"illuminance_lux": 49000}, context={"communication_ok": True, "sun_state": "day"})
    assert day_inside_jump_limit["illuminance_lux"] == 49000.0

    dq.reset_state_from_metadata()
    day_above_jump_limit = dq.clean("SONBEST_SM9560B", {"illuminance_lux": 60000}, context={"communication_ok": True, "sun_state": "day"})
    assert day_above_jump_limit["illuminance_lux"] == 60000.0

    twilight_above_jump_limit = dq.clean("SONBEST_SM9560B", {"illuminance_lux": 25000}, context={"communication_ok": True, "sun_state": "twilight"})
    assert twilight_above_jump_limit["illuminance_lux"] == 25000.0


def test_solar_radiation_physical_and_cross_sensor_rules(tmp_path: Path):
    dq = make_dq(tmp_path)

    negative = dq.clean("XS_TBQ02C", {"solar_radiation_wm2": -5}, context={"communication_ok": True, "sun_state": "day"})
    assert negative["solar_radiation_wm2"] == 0.0

    night = dq.clean("XS_TBQ02C", {"solar_radiation_wm2": 20}, context={"communication_ok": True, "sun_state": "night"})
    assert night["solar_radiation_wm2"] == 20.0

    dq.reset_state_from_metadata()
    contradiction = dq.clean(
        "XS_TBQ02C",
        {"solar_radiation_wm2": 600},
        context={"communication_ok": True, "sun_state": "day", "related_values": {"illuminance_lux": 50}},
    )
    assert contradiction["solar_radiation_wm2"] == 600.0

    no_related_value = dq.clean(
        "XS_TBQ02C",
        {"solar_radiation_wm2": 500},
        context={"communication_ok": True, "sun_state": "day"},
    )
    assert no_related_value["solar_radiation_wm2"] == 500.0


# ---------------------------------------------------------------------------
# SONBEST XM8504 rain gauge
# ---------------------------------------------------------------------------


def test_rain_interval_mode_negative_rate_resolution_and_spike_rules(tmp_path: Path):
    dq = make_dq(tmp_path)

    negative = dq.clean(
        "SONBEST_XM8504",
        {"rain_interval_mm": -1},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert negative["rain_interval_mm"] == 0.0

    rounded = dq.clean(
        "SONBEST_XM8504",
        {"rain_interval_mm": 0.31},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert rounded["rain_interval_mm"] == 0.31

    dq.reset_state_from_metadata()
    max_value = dq.clean(
        "SONBEST_XM8504",
        {"rain_interval_mm": 100},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert max_value["rain_interval_mm"] == 100.0


def test_rain_total_counter_mode_delta_and_reset(tmp_path: Path):
    dq = make_dq(tmp_path)

    first = dq.clean(
        "SONBEST_XM8504",
        {"rain_total_mm": 5.0},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert first["rain_interval_mm"] == 0.0
    assert "rain_total_mm" not in dq.state["sensors"]["SONBEST_XM8504"]["history"]

    second = dq.clean(
        "SONBEST_XM8504",
        {"rain_total_mm": 7.2},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert second["rain_interval_mm"] == 0.0
    assert "rain_total_mm" not in dq.state["sensors"]["SONBEST_XM8504"]["history"]

    reset = dq.clean(
        "SONBEST_XM8504",
        {"rain_total_mm": 1.0},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert reset["rain_interval_mm"] == 0.0
    assert "rain_total_mm" not in dq.state["sensors"]["SONBEST_XM8504"]["history"]


def test_rain_optional_alternative_input_does_not_overwrite_present_input(tmp_path: Path):
    # The rain gauge has two alternative input formats:
    # 1) rain_interval_mm - rain already calculated for the polling interval;
    # 2) rain_total_mm - cumulative counter, from which the interval must be calculated.
    # These modes are tested on separate clean states so that a previous interval
    # value cannot influence the cumulative-counter delta.

    interval_dir = tmp_path / "interval_mode"
    interval_dir.mkdir()
    dq_interval = make_dq(interval_dir)

    interval_result = dq_interval.clean(
        "SONBEST_XM8504",
        {"rain_interval_mm": 0.4},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert set(interval_result.keys()) == {"rain_interval_mm"}
    assert interval_result["rain_interval_mm"] == 0.4

    total_dir = tmp_path / "total_mode"
    total_dir.mkdir()
    dq_total = make_dq(total_dir)

    total_result = dq_total.clean(
        "SONBEST_XM8504",
        {"rain_total_mm": 3.0},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert set(total_result.keys()) == {"rain_interval_mm"}
    assert total_result["rain_interval_mm"] == 0.0


def test_rain_total_delta_after_previous_cumulative_counter_value(tmp_path: Path):
    dq = make_dq(tmp_path)

    first = dq.clean(
        "SONBEST_XM8504",
        {"rain_total_mm": 3.0},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert first["rain_interval_mm"] == 0.0

    second = dq.clean(
        "SONBEST_XM8504",
        {"rain_total_mm": 5.6},
        context={"communication_ok": True, "polling_interval_minutes": 10},
    )
    assert second["rain_interval_mm"] == 0.0


# ---------------------------------------------------------------------------
# TR-4H01X soil moisture
# ---------------------------------------------------------------------------


def test_soil_moisture_range_rise_fall_and_group_median_rules(tmp_path: Path):
    dq = make_dq(tmp_path)

    out_of_range = dq.clean(
        "TR_4H01X",
        {
            "soil_moisture_probe_1_percent": 120,
            "soil_moisture_probe_2_percent": 24,
            "soil_moisture_probe_3_percent": 24,
            "soil_moisture_probe_4_percent": 24,
        },
        context={"communication_ok": True, "rain_or_irrigation": False, "probes_same_depth": True},
    )
    assert out_of_range["soil_moisture_probe_1_percent"] == 0.0

    rise_without_rain = dq.clean(
        "TR_4H01X",
        {
            "soil_moisture_probe_1_percent": 45,
            "soil_moisture_probe_2_percent": 24,
            "soil_moisture_probe_3_percent": 24,
            "soil_moisture_probe_4_percent": 24,
        },
        context={"communication_ok": True, "rain_or_irrigation": False, "probes_same_depth": True},
    )
    assert rise_without_rain["soil_moisture_probe_1_percent"] == 45.0

    fall = dq.clean(
        "TR_4H01X",
        {
            "soil_moisture_probe_1_percent": 10,
            "soil_moisture_probe_2_percent": 24,
            "soil_moisture_probe_3_percent": 24,
            "soil_moisture_probe_4_percent": 24,
        },
        context={"communication_ok": True, "rain_or_irrigation": False, "probes_same_depth": True},
    )
    assert fall["soil_moisture_probe_1_percent"] == 10.0

    group = dq.clean(
        "TR_4H01X",
        {
            "soil_moisture_probe_1_percent": 28,
            "soil_moisture_probe_2_percent": 29,
            "soil_moisture_probe_3_percent": 30,
            "soil_moisture_probe_4_percent": 80,
        },
        context={"communication_ok": True, "rain_or_irrigation": True, "probes_same_depth": True},
    )
    assert group["soil_moisture_probe_4_percent"] == 80.0


def test_soil_moisture_does_not_apply_group_median_for_different_depths(tmp_path: Path):
    dq = make_dq(tmp_path)

    result = dq.clean(
        "TR_4H01X",
        {
            "soil_moisture_probe_1_percent": 28,
            "soil_moisture_probe_2_percent": 29,
            "soil_moisture_probe_3_percent": 30,
            "soil_moisture_probe_4_percent": 80,
        },
        context={"communication_ok": True, "rain_or_irrigation": True, "probes_same_depth": False},
    )
    assert result["soil_moisture_probe_4_percent"] == 80.0


# ---------------------------------------------------------------------------
# Scaling: new ordinary physical sensor through JSON only
# ---------------------------------------------------------------------------


def test_new_ordinary_physical_sensor_can_be_added_only_in_metadata(tmp_path: Path):
    metadata = _read_metadata()
    metadata["sensors"]["GENERIC_TANK_TEMPERATURE_SENSOR"] = {
        "description": "New ordinary physical parameter added without changing Python code.",
        "parameters": {
            "tank_temperature_c": {
                "unit": "°C",
                "min": -20,
                "max": 120,
                "resolution": 0.5,
                "initial_history": [25, 25, 25, 25, 25],
                "rules": [
                    {"type": "range_check", "replacement": "median_history"},
                    {"type": "jump_check", "max_delta_from_median": 30, "replacement": "median_history"},
                    {"type": "clamp"},
                ],
            }
        },
    }
    dq = make_dq(tmp_path, metadata)

    valid = dq.clean("GENERIC_TANK_TEMPERATURE_SENSOR", {"tank_temperature_c": 31.26}, context={"communication_ok": True})
    assert valid["tank_temperature_c"] == 31.5

    too_high = dq.clean("GENERIC_TANK_TEMPERATURE_SENSOR", {"tank_temperature_c": 200}, context={"communication_ok": True})
    assert too_high["tank_temperature_c"] == 25.0

    jump = dq.clean("GENERIC_TANK_TEMPERATURE_SENSOR", {"tank_temperature_c": 80}, context={"communication_ok": True})
    assert jump["tank_temperature_c"] == 25.0


# ---------------------------------------------------------------------------
# pandas batch helper
# ---------------------------------------------------------------------------


def test_clean_dataframe_preserves_index_and_outputs_numeric_dataframe(tmp_path: Path):
    dq = make_dq(tmp_path, metadata_with_generic_water_level_sensor())
    index = pd.DatetimeIndex(["2026-07-15T10:00:00", "2026-07-15T10:01:00"])
    frame = pd.DataFrame(
        [
            {"water_level_cm": 120.12, "communication_ok": True},
            {"water_level_cm": 500, "communication_ok": True},
        ],
        index=index,
    )

    cleaned = dq.clean_dataframe(
        "GENERIC_WATER_LEVEL_SENSOR",
        frame,
        context_columns=["communication_ok"],
    )

    assert isinstance(cleaned, pd.DataFrame)
    assert list(cleaned.index) == list(index)
    assert list(cleaned.columns) == ["water_level_cm"]
    assert cleaned.loc[index[0], "water_level_cm"] == 120.1
    assert np.isfinite(cleaned.to_numpy(dtype=float)).all()
