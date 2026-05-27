import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template

from algorithms.data_quality import DataQuality
from SensorAgent.sensor_agent import SENSOR_DATA_ONTOLOGY
from StatusAgent import AgentStatus, StatusAwareAgent, utc_now_iso

load_dotenv()

logger = logging.getLogger(__name__)

QUALITY_DATA_ONTOLOGY = "quality-data"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def sensor_input_template() -> Template:
    template = Template()
    template.set_metadata("ontology", SENSOR_DATA_ONTOLOGY)
    template.set_metadata("performative", "inform")
    return template


def current_sun_state() -> str:
    now = datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0

    if 6.0 <= hour < 8.0 or 18.0 <= hour < 20.0:
        return "twilight"

    if 8.0 <= hour < 18.0:
        return "day"

    return "night"


class SensorQualityReceiverBehaviour(CyclicBehaviour):
    async def on_start(self) -> None:
        logger.info("Data quality receiver started")

    async def run(self) -> None:
        msg = await self.receive(timeout=5)

        if msg is None:
            return

        try:
            body = json.loads(msg.body)
        except json.JSONDecodeError:
            self.agent.set_agent_status(
                AgentStatus.DEGRADED,
                "data quality received invalid JSON",
                priority=self.agent.priority,
            )
            logger.warning("Data quality agent received invalid JSON")
            return

        async with self.agent.state_lock:
            await self.agent.process_sensor_sample(body, self)


class DataQualityHealthBehaviour(PeriodicBehaviour):
    async def run(self) -> None:
        await self.agent.refresh_presence_status()


class DataQualityAgent(StatusAwareAgent):
    def __init__(self, jid: str, password: str, priority: int = 7, **kwargs):
        super().__init__(jid, password, **kwargs)
        self.priority = priority

    async def setup(self) -> None:
        logger.info("Data quality agent started: %s", self.jid)

        self.state_lock = asyncio.Lock()
        self.irrigator_agent_jid = self.get("irrigator_agent_jid") or os.getenv(
            "IRRIGATOR_AGENT_JID",
            "irrigator@localhost",
        )
        self.expected_sensor_names = self.parse_expected_sensor_names()

        self.latest_raw_by_sensor: dict[str, dict[str, Any]] = {}
        self.latest_clean_by_sensor: dict[str, dict[str, Any]] = {}
        self.latest_clean_values: dict[str, Any] = {}
        self.processed_count = 0
        self.last_processed_monotonic: float | None = None
        self.last_error: str | None = None

        metadata_path = os.getenv(
            "DATA_QUALITY_METADATA_FILE",
            "data_quality_metadata.json",
        )
        state_path = os.getenv(
            "DATA_QUALITY_MEMORY_FILE",
            "./data/data_quality_state.json",
        )
        self.data_quality = DataQuality(
            metadata_path=metadata_path,
            state_path=state_path,
            autosave=True,
        )

        self.output_file = Path(
            os.getenv("CLEAN_SENSOR_STATE_FILE", "./data/sensors_state.json")
        )
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            "data quality waiting for sensor data",
            priority=self.priority,
        )
        self.set_offline_message("stopped normally")

        self.add_behaviour(SensorQualityReceiverBehaviour(), sensor_input_template())

        period = env_float("DATA_QUALITY_HEALTH_CHECK_SECONDS", 5.0)
        self.add_behaviour(DataQualityHealthBehaviour(period=period))

    def parse_expected_sensor_names(self) -> list[str]:
        raw = os.getenv(
            "EXPECTED_SENSOR_NAMES",
            "Veinasa_THPW_NJ,SONBEST_SM9560B,XS_TBQ02C,SONBEST_XM8504,TR_4H01X",
        )
        return [item.strip() for item in raw.split(",") if item.strip()]

    async def process_sensor_sample(
        self,
        body: dict[str, Any],
        sender_behaviour: CyclicBehaviour,
    ) -> None:
        sensor_name = str(body.get("sensor_name") or "")
        sensor_id = str(body.get("sensor_id") or sensor_name)
        timestamp = str(body.get("timestamp") or utc_now_iso())
        values = body.get("values")

        if not sensor_name or not isinstance(values, dict):
            self.last_error = "sensor sample missing sensor_name or values"
            self.set_agent_status(
                AgentStatus.DEGRADED,
                self.last_error,
                priority=self.priority,
            )
            logger.warning("Invalid sensor sample for data quality: %s", body)
            return

        context = self.build_quality_context(sensor_name, values, body.get("context"))

        try:
            result = self.data_quality.clean(
                sensor_name=sensor_name,
                values=values,
                context=context,
                return_details=True,
            )
        except Exception as error:
            self.last_error = str(error)
            self.set_agent_status(
                AgentStatus.DEGRADED,
                f"data quality failed for {sensor_name}",
                priority=self.priority,
            )
            logger.exception("Data quality failed for %s", sensor_name)
            return

        cleaned_values = dict(result["values"])
        flags = dict(result["flags"])
        record = {
            "kind": "quality_sample",
            "sensor_id": sensor_id,
            "sensor_name": sensor_name,
            "source_name": body.get("source_name"),
            "timestamp": timestamp,
            "values": cleaned_values,
            "flags": flags,
            "raw_values": values,
            "context": context,
            "source_anomaly": bool(body.get("is_anomaly", False)),
            "source_anomaly_type": body.get("anomaly_type"),
            "processed_at": utc_now_iso(),
        }

        self.latest_raw_by_sensor[sensor_name] = body
        self.latest_clean_by_sensor[sensor_name] = record
        self.latest_clean_values.update(cleaned_values)
        self.processed_count += 1
        self.last_processed_monotonic = time.monotonic()
        self.last_error = None

        self.write_clean_state()
        await self.forward_to_irrigator(record, sender_behaviour)
        await self.refresh_presence_status()

        logger.info(
            "Data quality processed %s sample from %s",
            "anomalous" if record["source_anomaly"] else "normal",
            sensor_name,
        )

    def build_quality_context(
        self,
        sensor_name: str,
        values: dict[str, Any],
        raw_context: Any,
    ) -> dict[str, Any]:
        context = dict(raw_context) if isinstance(raw_context, dict) else {}
        context.setdefault("communication_ok", True)
        context.setdefault("sun_state", current_sun_state())
        context.setdefault("polling_interval_minutes", 10)
        context.setdefault("probes_same_depth", True)

        rain_value = self.latest_clean_values.get("rain_interval_mm")
        if rain_value is None:
            rain_value = values.get("rain_interval_mm")
        context["rain_or_irrigation"] = bool(float(rain_value or 0.0) > 0.0)

        related_values = dict(context.get("related_values") or {})
        related_values.update(self.latest_clean_values)
        context["related_values"] = related_values

        if sensor_name == "TR_4H01X":
            context.setdefault("probes_same_depth", True)

        return context

    async def forward_to_irrigator(
        self,
        record: dict[str, Any],
        sender_behaviour: CyclicBehaviour,
    ) -> None:
        msg = Message(to=self.irrigator_agent_jid)
        msg.set_metadata("performative", "inform")
        msg.set_metadata("ontology", QUALITY_DATA_ONTOLOGY)
        msg.set_metadata("kind", "quality_sample")
        msg.body = json.dumps(record, ensure_ascii=False)

        try:
            await sender_behaviour.send(msg)
        except Exception as error:
            self.last_error = str(error)
            self.set_agent_status(
                AgentStatus.DEGRADED,
                "data quality output forwarding failed",
                priority=self.priority,
            )
            logger.exception("Data quality failed to forward output")

    async def refresh_presence_status(self) -> None:
        if self.last_error:
            self.set_agent_status(
                AgentStatus.DEGRADED,
                "data quality error",
                priority=self.priority,
            )
            return

        seen = sum(
            1
            for sensor_name in self.expected_sensor_names
            if sensor_name in self.latest_clean_by_sensor
        )
        expected = len(self.expected_sensor_names)

        if seen < expected:
            status = AgentStatus.ONLINE_IDLE
            message = f"data quality waiting for sensors ({seen}/{expected})"
        else:
            status = AgentStatus.WORKING
            message = f"data quality cleaned samples ({seen}/{expected})"

        self.set_agent_status(status, message, priority=self.priority)
        self.set_offline_message(f"stopped normally; previous status {status.value}")

    def write_clean_state(self) -> None:
        payload = [
            self.latest_clean_by_sensor[sensor_name]
            for sensor_name in sorted(self.latest_clean_by_sensor)
        ]
        tmp_file = self.output_file.with_name(f".{self.output_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.output_file)
