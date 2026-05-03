import json
import logging
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.strip().replace(",", ".")

        if not value:
            return None

        try:
            return float(value)
        except ValueError:
            return None

    return None


def get_first_float(source: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in source:
            value = coerce_float(source[key])

            if value is not None:
                return value

    return None


@dataclass(frozen=True)
class SensorRecord:
    gateway_guid: str
    timestamp: str
    value: Any
    received_monotonic: float


@dataclass(frozen=True)
class WeatherInput:
    temperature_c: float
    relative_humidity: float
    pressure_kpa: float
    wind_speed_ms: float


@dataclass(frozen=True)
class RainInput:
    rainfall_mm: float
    is_raining: bool


@dataclass(frozen=True)
class IrrigationDecision:
    timestamp: str
    value: bool


@dataclass(frozen=True)
class IrrigationAlgorithmConfig:
    weather_gateway_guid: str
    solar_gateway_guid: str
    light_gateway_guid: str
    rain_gateway_guid: str

    crop_coefficient: float = 1.0
    management_allowed_depletion_mm: float = 25.0
    max_depletion_mm: float = 80.0
    max_wind_speed_ms: float = 8.0
    default_step_seconds: float = 300.0
    rain_value_mode: str = "incremental"
    memory_file: str = "./data/irrigation_memory.json"

    @classmethod
    def from_env(cls) -> "IrrigationAlgorithmConfig":
        return cls(
            weather_gateway_guid=env_str(
                "GATEWAY_WEATHER_GUID",
                "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
            ),
            solar_gateway_guid=env_str(
                "GATEWAY_SOLAR_GUID",
                "9f8e7d6c-5b4a-3f2e-1d0c-b9a8f7e6d5c4",
            ),
            light_gateway_guid=env_str(
                "GATEWAY_LIGHT_GUID",
                "8f3e2a1c-9b7d-4e5f-a1b2-c3d4e5f6a7b8",
            ),
            rain_gateway_guid=env_str(
                "GATEWAY_RAIN_GUID",
                "4b5c6d7e-8f9a-0b1c-2d3e-4f5a6b7c8d9e",
            ),
            crop_coefficient=env_float("IRRIGATION_CROP_COEFFICIENT", 1.0),
            management_allowed_depletion_mm=env_float("IRRIGATION_MAD_MM", 25.0),
            max_depletion_mm=env_float("IRRIGATION_MAX_DEPLETION_MM", 80.0),
            max_wind_speed_ms=env_float("IRRIGATION_MAX_WIND_SPEED_MS", 8.0),
            default_step_seconds=env_float("IRRIGATION_DEFAULT_STEP_SECONDS", 300.0),
            rain_value_mode=env_str("RAIN_VALUE_MODE", "incremental").lower(),
            memory_file=env_str(
                "IRRIGATION_MEMORY_FILE",
                "./data/irrigation_memory.json",
            ),
        )


class IrrigationAlgorithm:
    """
    Внешний алгоритм автополива.

    Агент передаёт сюда только нормализованные данные.
    Внутри алгоритм:
    1. Берёт данные метеостанции, солнечной радиации, освещённости и дождя.
    2. Рассчитывает приближённую ETo по Penman-Monteith.
    3. Обновляет расчётный дефицит воды в почве.
    4. Возвращает true/false: нужен полив или нет.
    """

    def __init__(self, config: IrrigationAlgorithmConfig | None = None):
        self.config = config or IrrigationAlgorithmConfig.from_env()
        self.memory_file = Path(self.config.memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        self._last_run_monotonic: float | None = None

    @property
    def required_gateway_guids(self) -> tuple[str, str, str, str]:
        return (
            self.config.weather_gateway_guid,
            self.config.solar_gateway_guid,
            self.config.light_gateway_guid,
            self.config.rain_gateway_guid,
        )

    def missing_gateways(
        self,
        records: Mapping[str, SensorRecord],
    ) -> list[str]:
        return [
            gateway_guid
            for gateway_guid in self.required_gateway_guids
            if gateway_guid not in records
        ]

    def stale_gateways(
        self,
        records: Mapping[str, SensorRecord],
        max_age_seconds: float,
    ) -> list[str]:
        now = time.monotonic()

        stale = []

        for gateway_guid in self.required_gateway_guids:
            record = records.get(gateway_guid)

            if record is None:
                stale.append(gateway_guid)
                continue

            if now - record.received_monotonic > max_age_seconds:
                stale.append(gateway_guid)

        return stale

    def run_from_records(
        self,
        records: Mapping[str, SensorRecord],
    ) -> IrrigationDecision:
        missing = self.missing_gateways(records)

        if missing:
            raise ValueError(f"Missing required gateway data: {missing}")

        weather = self._extract_weather(
            records[self.config.weather_gateway_guid].value
        )
        solar_radiation_wm2 = self._extract_solar_radiation(
            records[self.config.solar_gateway_guid].value
        )
        illuminance_lux = self._extract_illuminance(
            records[self.config.light_gateway_guid].value
        )
        rain = self._extract_rain(
            records[self.config.rain_gateway_guid].value
        )

        return self.run(
            weather=weather,
            solar_radiation_wm2=solar_radiation_wm2,
            illuminance_lux=illuminance_lux,
            rain=rain,
        )

    def run(
        self,
        weather: WeatherInput,
        solar_radiation_wm2: float,
        illuminance_lux: float,
        rain: RainInput,
    ) -> IrrigationDecision:
        memory = self._load_memory()

        now_iso = utc_now_iso()
        now_monotonic = time.monotonic()

        if self._last_run_monotonic is None:
            dt_hours = self.config.default_step_seconds / 3600.0
        else:
            dt_hours = max(
                0.0,
                (now_monotonic - self._last_run_monotonic) / 3600.0,
            )

        self._last_run_monotonic = now_monotonic

        eto_mm_hour = self._calculate_hourly_eto(
            temperature_c=weather.temperature_c,
            relative_humidity=weather.relative_humidity,
            pressure_kpa=weather.pressure_kpa,
            wind_speed_ms=weather.wind_speed_ms,
            solar_radiation_wm2=solar_radiation_wm2,
            illuminance_lux=illuminance_lux,
        )

        etc_mm = self.config.crop_coefficient * eto_mm_hour * dt_hours
        effective_rainfall_mm = self._get_effective_rainfall(memory, rain)

        depletion_mm = float(memory.get("depletion_mm", 0.0))
        depletion_mm = depletion_mm + etc_mm - effective_rainfall_mm
        depletion_mm = min(
            max(depletion_mm, 0.0),
            self.config.max_depletion_mm,
        )

        wind_too_high = weather.wind_speed_ms > self.config.max_wind_speed_ms

        irrigation_required = (
            depletion_mm >= self.config.management_allowed_depletion_mm
            and not rain.is_raining
            and not wind_too_high
        )

        memory.update(
            {
                "updated_at": now_iso,
                "depletion_mm": depletion_mm,
                "last_eto_mm_hour": eto_mm_hour,
                "last_etc_increment_mm": etc_mm,
                "last_effective_rainfall_mm": effective_rainfall_mm,
                "last_irrigation_required": irrigation_required,
                "last_weather": {
                    "temperature_c": weather.temperature_c,
                    "relative_humidity": weather.relative_humidity,
                    "pressure_kpa": weather.pressure_kpa,
                    "wind_speed_ms": weather.wind_speed_ms,
                },
                "last_solar_radiation_wm2": solar_radiation_wm2,
                "last_illuminance_lux": illuminance_lux,
                "last_rain": {
                    "rainfall_mm": rain.rainfall_mm,
                    "is_raining": rain.is_raining,
                },
            }
        )

        self._write_memory(memory)

        logger.info(
            "Irrigation decision=%s, depletion=%.2f mm, ETo=%.4f mm/h, ETc+=%.4f mm, rain=%.2f mm",
            irrigation_required,
            depletion_mm,
            eto_mm_hour,
            etc_mm,
            effective_rainfall_mm,
        )

        return IrrigationDecision(
            timestamp=now_iso,
            value=irrigation_required,
        )

    def _extract_weather(self, value: Any) -> WeatherInput:
        if not isinstance(value, dict):
            raise ValueError("Weather gateway value must be an object")

        temperature = get_first_float(
            value,
            ["temperature", "temperature_c", "air_temperature", "temp"],
        )
        humidity = get_first_float(
            value,
            ["humidity", "relative_humidity", "rh"],
        )
        pressure = get_first_float(
            value,
            ["pressure", "pressure_hpa", "pressure_kpa", "atmospheric_pressure"],
        )
        wind_speed = get_first_float(
            value,
            ["wind_speed", "wind_speed_ms", "windSpeed"],
        )

        if temperature is None:
            raise ValueError("Weather temperature is missing")

        if humidity is None:
            raise ValueError("Weather humidity is missing")

        if pressure is None:
            pressure = 101.3

        if wind_speed is None:
            wind_speed = 2.0

        return WeatherInput(
            temperature_c=temperature,
            relative_humidity=humidity,
            pressure_kpa=pressure,
            wind_speed_ms=wind_speed,
        )

    def _extract_solar_radiation(self, value: Any) -> float:
        if isinstance(value, dict):
            result = get_first_float(
                value,
                [
                    "solar_radiation",
                    "radiation",
                    "irradiance",
                    "global_radiation",
                    "value",
                ],
            )
        else:
            result = coerce_float(value)

        if result is None:
            raise ValueError("Solar radiation value is missing")

        return result

    def _extract_illuminance(self, value: Any) -> float:
        if isinstance(value, dict):
            result = get_first_float(
                value,
                ["illuminance", "lux", "light", "value"],
            )
        else:
            result = coerce_float(value)

        if result is None:
            return 0.0

        return result

    def _extract_rain(self, value: Any) -> RainInput:
        rainfall_mm = 0.0
        is_raining = False

        if isinstance(value, dict):
            rainfall_value = get_first_float(
                value,
                [
                    "rainfall_mm",
                    "rain_mm",
                    "precipitation_mm",
                    "precipitation",
                    "rainfall",
                    "value",
                ],
            )

            if rainfall_value is not None:
                rainfall_mm = max(0.0, rainfall_value)

            rain_flag = (
                value.get("is_raining")
                if "is_raining" in value
                else value.get("raining")
                if "raining" in value
                else value.get("rain_detected")
                if "rain_detected" in value
                else value.get("rain")
            )

            is_raining = self._parse_rain_flag(rain_flag)

            if rainfall_mm > 0:
                is_raining = True

        else:
            rainfall_value = coerce_float(value)

            if rainfall_value is not None:
                rainfall_mm = max(0.0, rainfall_value)
                is_raining = rainfall_mm > 0

        return RainInput(
            rainfall_mm=rainfall_mm,
            is_raining=is_raining,
        )

    def _parse_rain_flag(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return value > 0

        if isinstance(value, str):
            return value.strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "rain",
                "raining",
            }

        return False

    def _calculate_hourly_eto(
        self,
        temperature_c: float,
        relative_humidity: float,
        pressure_kpa: float,
        wind_speed_ms: float,
        solar_radiation_wm2: float,
        illuminance_lux: float,
    ) -> float:
        """
        Упрощённая почасовая FAO-56 Penman-Monteith.

        Входы:
        - temperature_c: °C
        - relative_humidity: %
        - pressure_kpa: kPa или hPa/Pa, нормализуется ниже
        - wind_speed_ms: m/s
        - solar_radiation_wm2: W/m²
        - illuminance_lux: lux

        Выход:
        - ETo, mm/hour
        """

        temperature_c = max(min(temperature_c, 60.0), -40.0)
        relative_humidity = max(min(relative_humidity, 100.0), 0.0)
        pressure_kpa = self._normalize_pressure_kpa(pressure_kpa)
        wind_speed_ms = max(wind_speed_ms, 0.0)
        solar_radiation_wm2 = max(solar_radiation_wm2, 0.0)
        illuminance_lux = max(illuminance_lux, 0.0)

        saturation_vapour_pressure = 0.6108 * math.exp(
            (17.27 * temperature_c) / (temperature_c + 237.3)
        )
        actual_vapour_pressure = saturation_vapour_pressure * relative_humidity / 100.0
        vapour_pressure_deficit = max(
            saturation_vapour_pressure - actual_vapour_pressure,
            0.0,
        )

        delta = (
            4098.0
            * saturation_vapour_pressure
            / ((temperature_c + 237.3) ** 2)
        )

        gamma = 0.000665 * pressure_kpa

        solar_radiation_mj_m2_hour = solar_radiation_wm2 * 3600.0 / 1_000_000.0

        is_daylight = solar_radiation_wm2 > 20.0 or illuminance_lux > 1000.0

        net_shortwave_radiation = 0.77 * solar_radiation_mj_m2_hour
        net_radiation = net_shortwave_radiation

        if is_daylight:
            soil_heat_flux = 0.1 * net_radiation
        else:
            soil_heat_flux = 0.0

        numerator = (
            0.408 * delta * (net_radiation - soil_heat_flux)
            + gamma
            * (37.0 / (temperature_c + 273.0))
            * wind_speed_ms
            * vapour_pressure_deficit
        )

        denominator = delta + gamma * (1.0 + 0.34 * wind_speed_ms)

        if denominator <= 0:
            return 0.0

        return max(numerator / denominator, 0.0)

    def _normalize_pressure_kpa(self, pressure: float) -> float:
        """
        Нормализация давления:
        - 101325  -> Pa  -> 101.325 kPa
        - 1013    -> hPa -> 101.3 kPa
        - 101.3   -> kPa -> 101.3 kPa
        """

        if pressure > 2000:
            return pressure / 1000.0

        if pressure > 200:
            return pressure / 10.0

        return pressure

    def _get_effective_rainfall(
        self,
        memory: dict[str, Any],
        rain: RainInput,
    ) -> float:
        if self.config.rain_value_mode == "cumulative":
            previous = memory.get("last_rain_cumulative_mm")
            current = rain.rainfall_mm

            memory["last_rain_cumulative_mm"] = current

            if previous is None:
                return 0.0

            return max(0.0, current - float(previous))

        return max(0.0, rain.rainfall_mm)

    def _load_memory(self) -> dict[str, Any]:
        if not self.memory_file.exists():
            return {}

        try:
            with self.memory_file.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            logger.warning("Irrigation memory file is corrupted. Reinitializing.")
        except OSError:
            logger.exception("Cannot read irrigation memory file")

        return {}

    def _write_memory(self, data: dict[str, Any]) -> None:
        tmp_file = self.memory_file.with_name(f".{self.memory_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.memory_file)