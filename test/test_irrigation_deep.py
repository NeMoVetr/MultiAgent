"""
Deep pytest tests for irrigation.py.

How to use:
    1. Put this file into the same folder as irrigation.py.
    2. Run:
       python -m pytest test_irrigation_deep.py -q

The test imports the production class directly:
    from irrigation import Irrigation

Network-dependent methods are monkeypatched in tests, so the unit tests do not
call real IP geocoding or Open-Elevation. pyet.pm_fao56 is also monkeypatched
in decision tests to make expected results deterministic.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from algorithms import Irrigation
from algorithms import irrigator


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_external_services(monkeypatch):
    """Disable real geocoding/elevation calls for object construction."""
    monkeypatch.setattr(Irrigation, "_Irrigation__get_location", staticmethod(lambda: (48.708, 44.514)))
    monkeypatch.setattr(Irrigation, "_Irrigation__get_elevation", staticmethod(lambda lat, lng: 100.0))
    return Irrigation


def patch_pyet_et0(monkeypatch, et0_value: float, calls: Dict[str, Any] | None = None) -> None:
    """Replace pyet.pm_fao56 with a deterministic function returning one ET0 value."""

    def fake_pm_fao56(tmean, wind, rs=None, rh=None, pressure=None, elevation=None, lat=None, **kwargs):
        if calls is not None:
            calls.clear()
            calls.update(
                {
                    "tmean": tmean,
                    "wind": wind,
                    "rs": rs,
                    "rh": rh,
                    "pressure": pressure,
                    "elevation": elevation,
                    "lat": lat,
                    "kwargs": kwargs,
                }
            )
        return pd.Series([et0_value], index=tmean.index)

    monkeypatch.setattr(irrigator.pyet, "pm_fao56", fake_pm_fao56)


def make_agent(**overrides) -> Irrigation:
    data = {
        "soil_raw": [14.0, 13.5, 14.2, 13.9],
        "T_mean": 30.1,
        "RH_mean": 33.0,
        "wind_speed": 1.8,
        "pressure_hpa": 1005.0,
        "solar_radiation_wm2": 760.0,
        "rain_mm": 0.0,
    }
    data.update(overrides)
    return Irrigation(**data)


def expected_decision_by_formula(
        soil_raw: list[float],
        et0: float,
        rain_mm: float,
        wind_speed: float,
) -> bool:
    """Mirror the calculation implemented inside Irrigation.__calculate()."""
    theta_fc = 0.24
    theta_wp = 0.0827
    root_depth_m = 0.5
    p_tab = 0.50
    crop_coefficient = 1.05
    max_wind = 5.0

    theta_values = np.array(soil_raw, dtype=float) / 100.0
    theta_avg = float(np.mean(theta_values))

    total_available_water = 1000 * (theta_fc - theta_wp) * root_depth_m
    root_zone_depletion = 1000 * (theta_fc - theta_avg) * root_depth_m
    root_zone_depletion = float(np.clip(root_zone_depletion, 0, total_available_water))

    crop_evapotranspiration = crop_coefficient * et0
    depletion_fraction = float(np.clip(p_tab + 0.04 * (5 - crop_evapotranspiration), 0.1, 0.8))
    readily_available_water = depletion_fraction * total_available_water

    return bool(
        root_zone_depletion >= readily_available_water
        and rain_mm <= 0
        and wind_speed <= max_wind
    )


# ---------------------------------------------------------------------------
# Constants and external wrappers
# ---------------------------------------------------------------------------


def test_private_constants_match_model_values():
    assert Irrigation._get_const("theta_fc") == pytest.approx(0.24)
    assert Irrigation._get_const("theta_wp") == pytest.approx(0.0827)
    assert Irrigation._get_const("Zr") == pytest.approx(0.5)
    assert Irrigation._get_const("p_tab") == pytest.approx(0.50)
    assert Irrigation._get_const("Kc") == pytest.approx(1.05)
    assert Irrigation._get_const("u_max") == pytest.approx(5.0)

    with pytest.raises(AttributeError):
        Irrigation._get_const("unknown_constant")


def test_get_location_success_and_failure(monkeypatch):
    monkeypatch.setattr(irrigator.geocoder, "ip", lambda _: type("Geo", (), {"ok": True, "lat": 48.7, "lng": 44.5})())
    assert Irrigation._Irrigation__get_location() == (48.7, 44.5)

    Irrigation._Irrigation__get_location._cache.clear()
    monkeypatch.setattr(irrigator.geocoder, "ip", lambda _: type("Geo", (), {"ok": False, "lat": None, "lng": None})())
    with pytest.raises(RuntimeError, match="coordinates"):
        Irrigation._Irrigation__get_location()


def test_get_elevation_success_and_failure(monkeypatch):
    class GoodResponse:
        status_code = 200

        def json(self):
            return {"results": [{"elevation": 123.4}]}

    calls: Dict[str, Any] = {}

    def fake_get(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return GoodResponse()

    monkeypatch.setattr(irrigator.requests, "get", fake_get)
    assert Irrigation._Irrigation__get_elevation(48.7, 44.5) == pytest.approx(123.4)
    assert "48.7,44.5" in calls["url"]
    assert calls["timeout"] == 10

    class BadResponse:
        status_code = 500

        def json(self):
            return {}

    Irrigation._Irrigation__get_elevation._cache.clear()
    monkeypatch.setattr(irrigator.requests, "get", lambda url, timeout: BadResponse())
    with pytest.raises(RuntimeError, match="height"):
        Irrigation._Irrigation__get_elevation(48.7, 44.5)


# ---------------------------------------------------------------------------
# pyet input preparation
# ---------------------------------------------------------------------------


def test_pyet_receives_prepared_series_and_converted_solar_radiation(monkeypatch, patched_external_services):
    calls: Dict[str, Any] = {}
    patch_pyet_et0(monkeypatch, et0_value=4.0, calls=calls)

    agent = make_agent(solar_radiation_wm2=760.0)
    assert isinstance(agent.get_decision(), bool)

    assert isinstance(calls["tmean"], pd.Series)
    assert isinstance(calls["wind"], pd.Series)
    assert isinstance(calls["rs"], pd.Series)
    assert isinstance(calls["rh"], pd.Series)

    expected_rs = 760.0 * 86400 / 1_000_000
    assert float(calls["rs"].iloc[0]) == pytest.approx(expected_rs)
    assert float(calls["tmean"].iloc[0]) == pytest.approx(30.1)
    assert float(calls["rh"].iloc[0]) == pytest.approx(33.0)
    assert float(calls["wind"].iloc[0]) == pytest.approx(1.8)
    assert calls["pressure"] == pytest.approx(1005.0)
    assert calls["elevation"] == pytest.approx(100.0)
    assert calls["lat"] == pytest.approx(48.708)


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_name, soil_raw, et0, rain_mm, wind_speed",
    [
        ("dry_soil_no_rain_low_wind_irrigate", [14.0, 13.5, 14.2, 13.9], 4.0, 0.0, 1.8),
        ("wet_soil_no_irrigation", [24.0, 24.0, 24.0, 24.0], 4.0, 0.0, 1.8),
        ("rain_blocks_irrigation", [14.0, 13.5, 14.2, 13.9], 4.0, 2.0, 1.8),
        ("high_wind_blocks_irrigation", [14.0, 13.5, 14.2, 13.9], 4.0, 0.0, 5.1),
        ("wind_equal_limit_allows_irrigation", [14.0, 13.5, 14.2, 13.9], 4.0, 0.0, 5.0),
        ("soil_above_field_capacity_no_irrigation", [30.0, 30.0, 30.0, 30.0], 4.0, 0.0, 1.8),
        ("very_dry_soil_depletion_clipped_to_taw", [0.0, 0.0, 0.0, 0.0], 4.0, 0.0, 1.8),
        ("high_et0_changes_raw_threshold", [20.0, 20.0, 20.0, 20.0], 20.0, 0.0, 1.8),
    ],
)
def test_decision_matches_formula(monkeypatch, patched_external_services, case_name, soil_raw, et0, rain_mm,
                                  wind_speed):
    patch_pyet_et0(monkeypatch, et0_value=et0)

    agent = make_agent(soil_raw=soil_raw, rain_mm=rain_mm, wind_speed=wind_speed)
    expected = expected_decision_by_formula(
        soil_raw=soil_raw,
        et0=et0,
        rain_mm=rain_mm,
        wind_speed=wind_speed,
    )
    assert agent.get_decision() is expected, case_name


def test_depletion_threshold_is_inclusive(monkeypatch, patched_external_services):
    et0 = 4.0
    patch_pyet_et0(monkeypatch, et0_value=et0)

    theta_fc = 0.24
    theta_wp = 0.0827
    root_depth_m = 0.5
    total_available_water = 1000 * (theta_fc - theta_wp) * root_depth_m
    crop_evapotranspiration = 1.05 * et0
    depletion_fraction = float(np.clip(0.50 + 0.04 * (5 - crop_evapotranspiration), 0.1, 0.8))
    readily_available_water = depletion_fraction * total_available_water
    theta_at_threshold = theta_fc - readily_available_water / (1000 * root_depth_m)
    soil_percent_at_threshold = theta_at_threshold * 100

    exactly_on_threshold = make_agent(soil_raw=[soil_percent_at_threshold] * 4)
    assert exactly_on_threshold.get_decision() is True

    slightly_wetter = make_agent(soil_raw=[soil_percent_at_threshold + 0.05] * 4)
    assert slightly_wetter.get_decision() is False


def test_nan_et0_results_in_no_irrigation(monkeypatch, patched_external_services):
    patch_pyet_et0(monkeypatch, et0_value=float("nan"))
    agent = make_agent()
    assert agent.get_decision() is False


# ---------------------------------------------------------------------------
# DataQuality output shape sent into Irrigation
# ---------------------------------------------------------------------------


def test_irrigation_accepts_values_from_data_quality_output(monkeypatch, patched_external_services):
    patch_pyet_et0(monkeypatch, et0_value=4.0)

    normalized_data = {
        "soil_moisture_probe_1_percent": 14.0,
        "soil_moisture_probe_2_percent": 13.5,
        "soil_moisture_probe_3_percent": 14.2,
        "soil_moisture_probe_4_percent": 13.9,
        "air_temperature_c": 30.1,
        "air_humidity_percent": 33.0,
        "wind_speed_ms": 1.8,
        "air_pressure_hpa": 1005.0,
        "solar_radiation_wm2": 760.0,
        "rain_interval_mm": 0.0,
    }

    agent = Irrigation(
        soil_raw=[
            normalized_data["soil_moisture_probe_1_percent"],
            normalized_data["soil_moisture_probe_2_percent"],
            normalized_data["soil_moisture_probe_3_percent"],
            normalized_data["soil_moisture_probe_4_percent"],
        ],
        T_mean=normalized_data["air_temperature_c"],
        RH_mean=normalized_data["air_humidity_percent"],
        wind_speed=normalized_data["wind_speed_ms"],
        pressure_hpa=normalized_data["air_pressure_hpa"],
        solar_radiation_wm2=normalized_data["solar_radiation_wm2"],
        rain_mm=normalized_data["rain_interval_mm"],
    )

    assert agent.get_decision() is True


def test_constructor_stores_input_values():
    agent = Irrigation(
        soil_raw=[15.0, 16.0, 17.0, 18.0],
        T_mean=29.0,
        RH_mean=40.0,
        wind_speed=2.0,
        pressure_hpa=1000.0,
        solar_radiation_wm2=500.0,
        rain_mm=1.0,

    )

    assert agent.soil_raw == [15.0, 16.0, 17.0, 18.0]
    assert agent.T_mean == pytest.approx(29.0)
    assert agent.RH_mean == pytest.approx(40.0)
    assert agent.wind_speed == pytest.approx(2.0)
    assert agent.pressure_hpa == pytest.approx(1000.0)
    assert agent.solar_radiation_wm2 == pytest.approx(500.0)
    assert agent.rain_mm == pytest.approx(1.0)
