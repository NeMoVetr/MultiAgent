import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import csv

import aiomqtt
from dotenv import load_dotenv
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.template import Template

from DataQualityAgent.data_quality_agent import QUALITY_DATA_ONTOLOGY
from StatusAgent import AgentStatus, StatusAwareAgent, utc_now_iso
from algorithms.irrigator import Irrigation

load_dotenv()

logger = logging.getLogger(__name__)

IRRIGATION_INPUT_COLUMNS = [
    "air_temperature_c",
    "air_humidity_percent",
    "air_pressure_hpa",
    "wind_speed_ms",
    "wind_direction_deg",
    "illuminance_lux",
    "solar_radiation_wm2",
    "rain_interval_mm",
    "soil_moisture_probe_1_percent",
    "soil_moisture_probe_2_percent",
    "soil_moisture_probe_3_percent",
    "soil_moisture_probe_4_percent",
]

IRRIGATION_CSV_COLUMNS = [
    "timestamp",
    "decided_at",
    "decision",
    *IRRIGATION_INPUT_COLUMNS,
    "inputs_json",
    "sensor_timestamps_json",
]

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def quality_input_template() -> Template:
    template = Template()
    template.set_metadata("ontology", QUALITY_DATA_ONTOLOGY)
    template.set_metadata("performative", "inform")
    return template


class IrrigatorReceiverBehaviour(CyclicBehaviour):
    async def on_start(self) -> None:
        logger.info("Irrigator receiver started")

    async def run(self) -> None:
        msg = await self.receive(timeout=5)

        if msg is None:
            return

        try:
            body = json.loads(msg.body)
        except json.JSONDecodeError:
            self.agent.set_agent_status(
                AgentStatus.DEGRADED,
                "irrigator received invalid JSON",
                priority=self.agent.priority,
            )
            logger.warning("Irrigator received invalid JSON")
            return

        async with self.agent.state_lock:
            await self.agent.process_quality_sample(body)


class IrrigatorHealthBehaviour(PeriodicBehaviour):
    async def run(self) -> None:
        await self.agent.refresh_presence_status()


class IrrigatorAgent(StatusAwareAgent):
    def __init__(self, jid: str, password: str, priority: int = 8, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.priority = priority

    async def setup(self) -> None:
        logger.info("Irrigator agent started: %s", self.jid)

        self.state_lock = asyncio.Lock()
        self.expected_sensor_names = self.parse_expected_sensor_names()
        self.latest_by_sensor: dict[str, dict[str, Any]] = {}
        self.latest_values: dict[str, float] = {}
        self.last_decision_at_monotonic: float | None = None
        self.last_error: str | None = None
        self.last_decision: bool | None = None

        self.output_file = Path(
            os.getenv("IRRIGATION_DECISION_FILE", "./data/irrigation_decision.json")
        )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.csv_output_file = Path(
            os.getenv(
                "IRRIGATION_DECISION_CSV_FILE",
                str(self.output_file.with_suffix(".csv")),
            )
        )
        self.csv_output_file.parent.mkdir(parents=True, exist_ok=True)

        self.mqtt_host = os.getenv("MQTT_HOST", "host.docker.internal")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
        self.mqtt_username = os.getenv("MQTT_USERNAME") or None
        self.mqtt_password = os.getenv("MQTT_PASSWORD") or None
        self.mqtt_keepalive = int(os.getenv("MQTT_KEEPALIVE_SECONDS", "60"))
        self.mqtt_qos = int(os.getenv("MQTT_QOS", "0"))
        self.mqtt_decision_topic = os.getenv(
            "MQTT_TOPIC_IRRIGATION_DECISION",
            "irrigation/decision",
        )

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            "irrigator waiting for clean sensor data",
            priority=self.priority,
        )
        self.set_offline_message("stopped normally")

        self.add_behaviour(IrrigatorReceiverBehaviour(), quality_input_template())

        period = env_float("IRRIGATOR_HEALTH_CHECK_SECONDS", 5.0)
        self.add_behaviour(IrrigatorHealthBehaviour(period=period))

    def parse_expected_sensor_names(self) -> list[str]:
        raw = os.getenv(
            "EXPECTED_SENSOR_NAMES",
            "Veinasa_THPW_NJ,SONBEST_SM9560B,XS_TBQ02C,SONBEST_XM8504,TR_4H01X",
        )
        return [item.strip() for item in raw.split(",") if item.strip()]

    async def process_quality_sample(self, body: dict[str, Any]) -> None:
        sensor_name = str(body.get("sensor_name") or "")
        values = body.get("values")

        if not sensor_name or not isinstance(values, dict):
            self.last_error = "quality sample missing sensor_name or values"
            self.set_agent_status(
                AgentStatus.DEGRADED,
                self.last_error,
                priority=self.priority,
            )
            logger.warning("Invalid quality sample for irrigator: %s", body)
            return

        self.latest_by_sensor[sensor_name] = body
        self.latest_values.update(
            {
                key: float(value)
                for key, value in values.items()
                if isinstance(value, (int, float))
            }
        )
        self.last_error = None

        if not self.has_complete_input():
            await self.refresh_presence_status()
            return

        try:
            decision = self.calculate_decision()
        except Exception as error:
            self.last_error = str(error)
            self.set_agent_status(
                AgentStatus.DEGRADED,
                "irrigator calculation failed",
                priority=self.priority,
            )
            logger.exception("Irrigator calculation failed")
            return

        self.last_decision = decision
        self.last_decision_at_monotonic = time.monotonic()
        payload = self.build_decision_payload(decision)
        self.write_decision(payload)
        self.append_decision_csv(payload)

        try:
            await self.publish_decision_mqtt(payload)
        except aiomqtt.MqttError as error:
            self.last_error = str(error)
            self.set_agent_status(
                AgentStatus.DEGRADED,
                "irrigator MQTT publish failed",
                priority=self.priority,
            )
            logger.warning(
                "Irrigator failed to publish decision to MQTT topic %s: %s",
                getattr(self, "mqtt_decision_topic", None),
                error,
            )
            return

        self.set_agent_status(
            AgentStatus.WORKING,
            f"irrigation decision={decision}",
            priority=self.priority,
        )
        self.set_offline_message("stopped normally; previous status WORKING")

        logger.info("Irrigator decision calculated: %s", payload)

    def has_complete_input(self) -> bool:
        return all(
            sensor_name in self.latest_by_sensor
            for sensor_name in self.expected_sensor_names
        ) and all(
            key in self.latest_values
            for key in [
                "soil_moisture_probe_1_percent",
                "soil_moisture_probe_2_percent",
                "soil_moisture_probe_3_percent",
                "soil_moisture_probe_4_percent",
                "air_temperature_c",
                "air_humidity_percent",
                "wind_speed_ms",
                "air_pressure_hpa",
                "solar_radiation_wm2",
                "rain_interval_mm",
            ]
        )

    def calculate_decision(self) -> bool:
        values = self.latest_values
        irrigation = Irrigation(
            soil_raw=[
                values["soil_moisture_probe_1_percent"],
                values["soil_moisture_probe_2_percent"],
                values["soil_moisture_probe_3_percent"],
                values["soil_moisture_probe_4_percent"],
            ],
            T_mean=values["air_temperature_c"],
            RH_mean=values["air_humidity_percent"],
            wind_speed=values["wind_speed_ms"],
            pressure_hpa=values["air_pressure_hpa"],
            solar_radiation_wm2=values["solar_radiation_wm2"],
            rain_mm=values["rain_interval_mm"],

        )
        return bool(irrigation.get_decision())

    def build_decision_payload(self, decision: bool) -> dict[str, Any]:
        timestamps = [
            str(record.get("timestamp"))
            for record in self.latest_by_sensor.values()
            if record.get("timestamp")
        ]
        timestamp = max(timestamps) if timestamps else utc_now_iso()

        return {
            "timestamp": timestamp,
            "value": decision,
            "decided_at": utc_now_iso(),
            "inputs": {
                key: self.latest_values[key]
                for key in sorted(self.latest_values)
            },
            "sensor_timestamps": {
                sensor_name: record.get("timestamp")
                for sensor_name, record in sorted(self.latest_by_sensor.items())
            },
        }

    async def refresh_presence_status(self) -> None:
        if self.last_error:
            self.set_agent_status(
                AgentStatus.DEGRADED,
                "irrigator error",
                priority=self.priority,
            )
            return

        if self.has_complete_input():
            status = AgentStatus.WORKING
            if self.last_decision is None:
                message = "irrigator has complete input"
            else:
                message = f"irrigation decision={self.last_decision}"
        else:
            seen = sum(
                1
                for sensor_name in self.expected_sensor_names
                if sensor_name in self.latest_by_sensor
            )
            expected = len(self.expected_sensor_names)
            status = AgentStatus.ONLINE_IDLE
            message = f"irrigator waiting for clean data ({seen}/{expected})"

        self.set_agent_status(status, message, priority=self.priority)
        self.set_offline_message(f"stopped normally; previous status {status.value}")

    def write_decision(self, payload: dict[str, Any]) -> None:
        tmp_file = self.output_file.with_name(f".{self.output_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.output_file)

    async def publish_decision_mqtt(self, payload: dict[str, Any]) -> None:
        topic = getattr(self, "mqtt_decision_topic", None)
        if not topic:
            return

        client_kwargs: dict[str, Any] = {
            "hostname": getattr(
                self,
                "mqtt_host",
                os.getenv("MQTT_HOST", "host.docker.internal"),
            ),
            "port": getattr(self, "mqtt_port", int(os.getenv("MQTT_PORT", "1883"))),
            "keepalive": getattr(
                self,
                "mqtt_keepalive",
                int(os.getenv("MQTT_KEEPALIVE_SECONDS", "60")),
            ),
        }

        username = getattr(self, "mqtt_username", os.getenv("MQTT_USERNAME") or None)
        password = getattr(self, "mqtt_password", os.getenv("MQTT_PASSWORD") or None)

        if username:
            client_kwargs["username"] = username
        if password:
            client_kwargs["password"] = password

        encoded_payload = json.dumps(payload, ensure_ascii=False)
        qos = getattr(self, "mqtt_qos", int(os.getenv("MQTT_QOS", "0")))

        async with aiomqtt.Client(**client_kwargs) as client:
            await client.publish(topic, encoded_payload, qos=qos)

        logger.info("Published irrigation decision to MQTT topic %s", topic)

    def append_decision_csv(self, payload: dict[str, Any]) -> None:
        inputs = dict(payload.get("inputs") or {})
        csv_file = getattr(
            self,
            "csv_output_file",
            self.output_file.with_suffix(".csv"),
        )
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "timestamp": payload.get("timestamp"),
            "decided_at": payload.get("decided_at"),
            "decision": payload.get("value"),
            "inputs_json": json.dumps(inputs, ensure_ascii=False, sort_keys=True),
            "sensor_timestamps_json": json.dumps(
                payload.get("sensor_timestamps") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

        row.update({
            column: inputs.get(column)
            for column in IRRIGATION_INPUT_COLUMNS
        })

        write_header = not csv_file.exists() or csv_file.stat().st_size == 0

        with csv_file.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=IRRIGATION_CSV_COLUMNS,
                extrasaction="ignore",
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)
            file.flush()
            os.fsync(file.fileno())
