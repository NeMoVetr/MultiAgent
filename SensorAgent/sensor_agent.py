import asyncio
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from spade.behaviour import PeriodicBehaviour
from spade.message import Message

from StatusAgent import AgentStatus, StatusAwareAgent, utc_now_iso

load_dotenv()

logger = logging.getLogger(__name__)

SENSOR_DATA_ONTOLOGY = "sensor-data"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def utc_hour_fraction() -> float:
    now = datetime.now(timezone.utc)
    return now.hour + now.minute / 60.0 + now.second / 3600.0


def sun_state() -> str:
    hour = utc_hour_fraction()

    if 6.0 <= hour < 8.0 or 18.0 <= hour < 20.0:
        return "twilight"

    if 8.0 <= hour < 18.0:
        return "day"

    return "night"


def day_curve() -> float:
    hour = utc_hour_fraction()
    value = math.sin((hour - 6.0) / 12.0 * math.pi)
    return max(0.0, value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class SensorSimulationResult:
    values: dict[str, Any]
    anomaly: bool = False
    anomaly_type: str | None = None


class SensorAgentBase(StatusAwareAgent):
    async def setup_sensor_agent(
        self,
        sensor_id: str,
        sensor_name: str,
        source_name: str,
    ) -> None:
        self.sensor_id = sensor_id
        self.sensor_name = sensor_name
        self.source_name = source_name
        self.data_quality_agent_jid = self.get("data_quality_agent_jid") or os.getenv(
            "DATA_QUALITY_AGENT_JID",
            "data_quality@localhost",
        )

        self.status_lock = asyncio.Lock()
        self.last_generated_at: str | None = None
        self.last_generated_monotonic: float | None = None
        self.last_send_error: str | None = None
        self.generated_count = 0
        self.anomaly_count = 0
        self.random = random.Random(f"{sensor_id}-{os.getpid()}-{time.time_ns()}")

        self.current_operational_status = AgentStatus.ONLINE_IDLE.value
        self.current_status_message = f"{self.source_name} simulator starting"

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            self.current_status_message,
            priority=5,
        )
        self.set_offline_message("stopped normally")

        period = env_float("SENSOR_SIMULATION_PERIOD_SECONDS", 5.0)
        self.add_behaviour(SensorSimulationBehaviour(period=period))

        logger.info(
            "Sensor simulator configured: sensor_id=%s, source=%s, data_quality=%s",
            self.sensor_id,
            self.source_name,
            self.data_quality_agent_jid,
        )

    def should_emit_anomaly(self) -> bool:
        probability = env_float("SENSOR_ANOMALY_PROBABILITY", 0.12)
        return self.random.random() < probability

    async def record_generation(
        self,
        result: SensorSimulationResult,
        sent_ok: bool,
        error: str | None = None,
    ) -> None:
        async with self.status_lock:
            self.generated_count += 1
            if result.anomaly:
                self.anomaly_count += 1

            self.last_generated_at = utc_now_iso()
            self.last_generated_monotonic = time.monotonic()
            self.last_send_error = error

            if sent_ok:
                status = AgentStatus.WORKING
                if result.anomaly:
                    message = (
                        f"{self.source_name} generated anomalous sample "
                        f"({self.anomaly_count}/{self.generated_count})"
                    )
                else:
                    message = f"{self.source_name} simulating sensor data"
            else:
                status = AgentStatus.DEGRADED
                message = f"{self.source_name} failed to send data"

            self.current_operational_status = status.value
            self.current_status_message = message
            self.set_offline_message(f"stopped normally; previous status {status.value}")

        self.set_agent_status(status, message, priority=5)

        if status == AgentStatus.DEGRADED:
            logger.warning(
                "Sensor %s status changed to %s: %s; error=%s",
                self.sensor_id,
                status.value,
                message,
                error,
            )

    def build_context(self, result: SensorSimulationResult) -> dict[str, Any]:
        period_seconds = env_float("SENSOR_SIMULATION_PERIOD_SECONDS", 5.0)
        context: dict[str, Any] = {
            "communication_ok": True,
            "sun_state": sun_state(),
            "polling_interval_minutes": max(period_seconds / 60.0, 1.0),
        }

        if self.sensor_name == "TR_4H01X":
            context["probes_same_depth"] = True
            context["rain_or_irrigation"] = False

        if self.sensor_name == "SONBEST_XM8504":
            context["rain_or_irrigation"] = (
                float(result.values.get("rain_interval_mm") or 0.0) > 0.0
            )

        return context

    def simulate_values(self) -> SensorSimulationResult:
        raise NotImplementedError


class SensorSimulationBehaviour(PeriodicBehaviour):
    async def on_start(self) -> None:
        logger.info("Sensor simulation started: %s", self.agent.sensor_id)

    async def run(self) -> None:
        result = self.agent.simulate_values()
        timestamp = utc_now_iso()
        body = {
            "kind": "sensor_sample",
            "sensor_id": self.agent.sensor_id,
            "sensor_name": self.agent.sensor_name,
            "source_name": self.agent.source_name,
            "timestamp": timestamp,
            "values": result.values,
            "context": self.agent.build_context(result),
            "is_anomaly": result.anomaly,
            "anomaly_type": result.anomaly_type,
            "sent_at": utc_now_iso(),
        }

        msg = Message(to=self.agent.data_quality_agent_jid)
        msg.set_metadata("performative", "inform")
        msg.set_metadata("ontology", SENSOR_DATA_ONTOLOGY)
        msg.set_metadata("kind", "sensor_sample")
        msg.body = json.dumps(body, ensure_ascii=False)

        try:
            await self.send(msg)
        except Exception as error:
            await self.agent.record_generation(result, sent_ok=False, error=str(error))
            logger.exception("Sensor %s failed to send sample", self.agent.sensor_id)
            return

        await self.agent.record_generation(result, sent_ok=True)
        logger.debug("Sensor %s generated sample: %s", self.agent.sensor_id, body)


class SM9560BAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="sm9560b",
            sensor_name="SONBEST_SM9560B",
            source_name="SONBEST SM9560B",
        )

    def simulate_values(self) -> SensorSimulationResult:
        anomaly = self.should_emit_anomaly()
        state = sun_state()
        curve = day_curve()

        if anomaly:
            if state == "night":
                return SensorSimulationResult(
                    {"illuminance_lux": self.random.uniform(8000, 20000)},
                    anomaly=True,
                    anomaly_type="night_illuminance",
                )
            return SensorSimulationResult(
                {"illuminance_lux": self.random.choice([-50, 90000, None])},
                anomaly=True,
                anomaly_type="out_of_range_or_missing",
            )

        if state == "night":
            value = self.random.uniform(0, 2)
        elif state == "twilight":
            value = self.random.uniform(500, 12000)
        else:
            value = 15000 + curve * 45000 + self.random.uniform(-1500, 1500)

        return SensorSimulationResult({"illuminance_lux": round(clamp(value, 0, 65535), 1)})


class THPWNJAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="thpwnj",
            sensor_name="Veinasa_THPW_NJ",
            source_name="Veinasa THPW-NJ",
        )

    def simulate_values(self) -> SensorSimulationResult:
        anomaly = self.should_emit_anomaly()
        curve = day_curve()

        if anomaly:
            return SensorSimulationResult(
                {
                    "air_temperature_c": self.random.choice([-80, 130]),
                    "air_humidity_percent": self.random.choice([-15, 140]),
                    "air_pressure_hpa": self.random.choice([200, 1300]),
                    "wind_speed_ms": self.random.choice([-3, 80]),
                    "wind_direction_deg": self.random.choice([-40, 720, None]),
                },
                anomaly=True,
                anomaly_type="weather_out_of_range",
            )

        temperature = 18 + curve * 10 + self.random.uniform(-1.5, 1.5)
        humidity = 62 - curve * 22 + self.random.uniform(-4, 4)
        pressure = 1012 + self.random.uniform(-4, 4)
        wind_speed = max(0.0, self.random.gauss(2.2, 1.0))
        wind_direction = self.random.uniform(0, 359)

        return SensorSimulationResult(
            {
                "air_temperature_c": round(clamp(temperature, -40, 80), 1),
                "air_humidity_percent": round(clamp(humidity, 0, 100), 1),
                "air_pressure_hpa": round(clamp(pressure, 300, 1100), 1),
                "wind_speed_ms": round(clamp(wind_speed, 0, 45), 1),
                "wind_direction_deg": round(clamp(wind_direction, 0, 359), 0),
            }
        )


class TBQ02CAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="tbq02c",
            sensor_name="XS_TBQ02C",
            source_name="XS-TBQ02C Total Radiation Sensor",
        )

    def simulate_values(self) -> SensorSimulationResult:
        anomaly = self.should_emit_anomaly()
        state = sun_state()
        curve = day_curve()

        if anomaly:
            value = self.random.choice([-25, 2600, 600 if state == "night" else None])
            return SensorSimulationResult(
                {"solar_radiation_wm2": value},
                anomaly=True,
                anomaly_type="solar_out_of_range_or_context",
            )

        if state == "night":
            value = self.random.uniform(0, 1)
        elif state == "twilight":
            value = self.random.uniform(10, 180)
        else:
            value = 180 + curve * 850 + self.random.uniform(-30, 30)

        return SensorSimulationResult({"solar_radiation_wm2": round(clamp(value, 0, 2000), 1)})


class XM8504Agent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="xm8504",
            sensor_name="SONBEST_XM8504",
            source_name="SONBEST XM8504",
        )

    def simulate_values(self) -> SensorSimulationResult:
        anomaly = self.should_emit_anomaly()

        if anomaly:
            return SensorSimulationResult(
                {"rain_interval_mm": self.random.choice([-1.0, 12000.0, 120.0])},
                anomaly=True,
                anomaly_type="rain_out_of_range_or_spike",
            )

        raining = self.random.random() < env_float("SENSOR_RAIN_PROBABILITY", 0.18)
        value = self.random.uniform(0.2, 6.0) if raining else 0.0
        return SensorSimulationResult({"rain_interval_mm": round(value, 1)})


class TR4H01XAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="tr4h01x",
            sensor_name="TR_4H01X",
            source_name="TR-4H01X RS485 Smart 4 Probe Soil Moisture Sensor",
        )

    def simulate_values(self) -> SensorSimulationResult:
        anomaly = self.should_emit_anomaly()
        base = self.random.uniform(18.0, 32.0)
        values = {
            f"soil_moisture_probe_{index}_percent": round(
                clamp(base + self.random.uniform(-1.5, 1.5), 0, 100),
                1,
            )
            for index in range(1, 5)
        }

        if anomaly:
            probe = self.random.randint(1, 4)
            values[f"soil_moisture_probe_{probe}_percent"] = self.random.choice(
                [-10.0, 140.0, 85.0, None]
            )
            return SensorSimulationResult(
                values,
                anomaly=True,
                anomaly_type="soil_probe_outlier",
            )

        return SensorSimulationResult(values)
