import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import aiomqtt
from spade.behaviour import OneShotBehaviour
from spade.message import Message

from StatusAgent import AgentStatus, StatusAwareAgent

logger = logging.getLogger(__name__)

SENSOR_DATA_ONTOLOGY = "sensor-data"

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


class MQTTSensorAgent(StatusAwareAgent):
    """Base class for concrete MQTT -> XMPP sensor agents."""

    SENSOR_NAME = "MQTT_SENSOR"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_SENSOR"
    DEFAULT_MQTT_TOPIC = "rs485/sensor"
    MEASUREMENT_KEYS: tuple[str, ...] = ()

    class MQTTSubscribeBehaviour(OneShotBehaviour):
        async def run(self) -> None:
            await self.agent.listen_mqtt_forever(self)

    async def setup(self) -> None:
        self.set("mqtt_host", os.getenv("MQTT_HOST", "host.docker.internal"))
        self.set("mqtt_port", int(os.getenv("MQTT_PORT", "1883")))
        self.set("mqtt_username", os.getenv("MQTT_USERNAME") or None)
        self.set("mqtt_password", os.getenv("MQTT_PASSWORD") or None)
        self.set("mqtt_keepalive", int(os.getenv("MQTT_KEEPALIVE_SECONDS", "60")))
        self.set("mqtt_qos", int(os.getenv("MQTT_QOS", "0")))
        self.set("mqtt_reconnect_seconds", float(os.getenv("MQTT_RECONNECT_SECONDS", "5")))
        self.set(
            "mqtt_no_data_warning_seconds",
            float(os.getenv("MQTT_NO_DATA_WARNING_SECONDS", "30")),
        )
        self.set("mqtt_topic", os.getenv(self.MQTT_TOPIC_ENV, self.DEFAULT_MQTT_TOPIC))
        self.set("mqtt_connected", False)
        self.set("last_mqtt_message_monotonic", None)

        self.set_offline_message("MQTT sensor agent stopped")
        self.set_sensor_status(
            AgentStatus.ONLINE_IDLE,
            f"starting MQTT listener for topic {self.get('mqtt_topic')}",
            priority=4,
        )

        logger.info(
            "%s configured for MQTT topic %s at %s:%s",
            self.SENSOR_NAME,
            self.get("mqtt_topic"),
            self.get("mqtt_host"),
            self.get("mqtt_port"),
        )

        self.add_behaviour(self.MQTTSubscribeBehaviour())

    def set_sensor_status(
            self,
            status: AgentStatus,
            message: str,
            priority: int = 5,
    ) -> None:
        try:
            self.set_agent_status(status, message, priority=priority)
        except Exception as exc:
            logger.debug("%s presence update failed: %s", self.SENSOR_NAME, exc)

    async def listen_mqtt_forever(self, behaviour: OneShotBehaviour) -> None:
        while not behaviour.is_killed():
            try:
                await self.listen_mqtt_once(behaviour)
            except asyncio.CancelledError:
                raise
            except aiomqtt.MqttError as exc:
                self.set("mqtt_connected", False)
                self.set_sensor_status(
                    AgentStatus.DEGRADED,
                    f"MQTT broker unavailable at {self.get('mqtt_host')}:{self.get('mqtt_port')}; reconnecting",
                    priority=3,
                )
                logger.warning(
                    "%s MQTT broker unavailable at %s:%s; reconnect in %.1f seconds: %s",
                    self.SENSOR_NAME,
                    self.get("mqtt_host"),
                    self.get("mqtt_port"),
                    float(self.get("mqtt_reconnect_seconds")),
                    exc,
                )
                await asyncio.sleep(float(self.get("mqtt_reconnect_seconds")))
            except Exception as exc:
                self.set("mqtt_connected", False)
                self.set_sensor_status(
                    AgentStatus.DEGRADED,
                    "MQTT listener failed; reconnecting",
                    priority=3,
                )
                logger.warning(
                    "%s MQTT listener failed; reconnect in %.1f seconds: %s",
                    self.SENSOR_NAME,
                    float(self.get("mqtt_reconnect_seconds")),
                    exc,
                )
                await asyncio.sleep(float(self.get("mqtt_reconnect_seconds")))

    async def listen_mqtt_once(self, behaviour: OneShotBehaviour) -> None:
        client_kwargs: dict[str, Any] = {
            "hostname": self.get("mqtt_host"),
            "port": self.get("mqtt_port"),
            "keepalive": self.get("mqtt_keepalive"),
        }

        if self.get("mqtt_username"):
            client_kwargs["username"] = self.get("mqtt_username")
        if self.get("mqtt_password"):
            client_kwargs["password"] = self.get("mqtt_password")

        topic = self.get("mqtt_topic")
        qos = self.get("mqtt_qos")

        async with aiomqtt.Client(**client_kwargs) as client:
            await client.subscribe(topic, qos=qos)
            self.set("mqtt_connected", True)
            self.set("last_mqtt_message_monotonic", None)
            self.set_sensor_status(
                AgentStatus.ONLINE_IDLE,
                f"MQTT connected and subscribed to {topic}; waiting for data",
                priority=4,
            )
            logger.info("%s connected to MQTT and subscribed to %s", self.SENSOR_NAME, topic)

            no_data_task = asyncio.create_task(self.warn_if_no_data(behaviour, topic))

            try:
                messages = client.messages
                if callable(messages):
                    messages = messages()

                async for message in messages:
                    if behaviour.is_killed():
                        break
                    await self.handle_mqtt_message(behaviour, message)
            finally:
                no_data_task.cancel()
                await asyncio.gather(no_data_task, return_exceptions=True)
                self.set("mqtt_connected", False)

    async def warn_if_no_data(self, behaviour: OneShotBehaviour, topic: str) -> None:
        threshold_seconds = self.get("mqtt_no_data_warning_seconds")

        while not behaviour.is_killed():
            await asyncio.sleep(threshold_seconds)

            if not self.get("mqtt_connected"):
                continue

            last_message_at = self.get("last_mqtt_message_monotonic")
            if last_message_at is None:
                message = f"MQTT subscribed to {topic}, but no data has been received yet"
            else:
                elapsed_seconds = time.monotonic() - last_message_at
                if elapsed_seconds < threshold_seconds:
                    continue
                message = (
                    f"MQTT subscribed to {topic}, but no data has been received "
                    f"for {elapsed_seconds:.0f} seconds"
                )

            self.set_sensor_status(AgentStatus.ONLINE_IDLE, message, priority=4)
            logger.warning("%s %s", self.SENSOR_NAME, message)

    async def handle_mqtt_message(self,behaviour: OneShotBehaviour,  message: Any) -> None:
        topic = str(getattr(getattr(message, "topic", self.get("mqtt_topic")), "value", self.get("mqtt_topic")))

        try:
            normalized = self.normalize_mqtt_payload(topic, message.payload)
        except Exception as exc:
            self.set_sensor_status(
                AgentStatus.DEGRADED,
                f"invalid MQTT payload from {topic}",
                priority=3,
            )
            logger.warning("%s invalid MQTT payload from %s: %s", self.SENSOR_NAME, topic, exc)
            return

        measurements = normalized["measurements"]
        if not measurements:
            self.set_sensor_status(
                AgentStatus.DEGRADED,
                f"MQTT payload from {topic} has no expected measurements",
                priority=3,
            )
            logger.warning(
                "%s MQTT payload from %s has no expected measurements",
                self.SENSOR_NAME,
                topic,
            )
            return

        await self.send_to_data_quality_agent(behaviour, normalized)
        self.set("last_mqtt_message_monotonic", time.monotonic())
        self.set_sensor_status(
            AgentStatus.WORKING,
            f"received MQTT data from {topic}: {', '.join(measurements)}",
            priority=6,
        )

    def normalize_mqtt_payload(self, topic: str, raw_payload: bytes) -> dict[str, Any]:
        payload = json.loads(raw_payload.decode("utf-8"))
        registers = payload["registers"]

        if not isinstance(registers, dict):
            raise ValueError("payload.registers must be an object")

        measurements = {
            key: registers[key]
            for key in self.MEASUREMENT_KEYS
            if key in registers
        }

        sensor_id = payload.get("sensorId")
        timestamp = payload.get("timestamp") or now_utc_iso()

        return {
            "type": "sensor_data",
            "source": "mqtt",
            "topic": topic,
            "sensor_id": sensor_id,
            "sensorId": sensor_id,
            "sensor_name": self.SENSOR_NAME,
            "sensorName": self.SENSOR_NAME,
            "timestamp": timestamp,
            "registers": registers,
            "measurements": measurements,
            "readings": measurements,
            "values": measurements,
            "data": measurements,
            "raw": payload,
        }

    async def send_to_data_quality_agent(self,  behaviour: OneShotBehaviour, normalized: dict[str, Any]) -> None:
        receiver = self.get("data_quality_agent_jid")
        if not receiver:
            raise RuntimeError("data_quality_agent_jid is not set")

        msg = Message(to=receiver)
        msg.set_metadata("performative", "inform")
        msg.set_metadata("ontology", "sensor-data")
        msg.set_metadata("content-type", "application/json")
        msg.body = json.dumps(normalized, ensure_ascii=False)

        await behaviour.send(msg)
        logger.info(
            "Forwarded MQTT payload from topic=%s sensor=%s fields=%s",
            normalized["topic"],
            normalized["sensor_name"],
            list(normalized["measurements"].keys()),
        )


class SM9560BAgent(MQTTSensorAgent):
    SENSOR_NAME = "SONBEST_SM9560B"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_ILLUMINANCE_SM9560B"
    DEFAULT_MQTT_TOPIC = "rs485/sm9560b"
    MEASUREMENT_KEYS = ("illuminance_lux",)


class THPWNJAgent(MQTTSensorAgent):
    SENSOR_NAME = "Veinasa_THPW_NJ"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_WEATHER_THPWNJ"
    DEFAULT_MQTT_TOPIC = "rs485/thpwnj"
    MEASUREMENT_KEYS = (
        "wind_speed_ms",
        "wind_direction_deg",
        "air_temperature_c",
        "air_humidity_percent",
        "air_pressure_hpa",
    )


class TBQ02CAgent(MQTTSensorAgent):
    SENSOR_NAME = "XS_TBQ02C"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_SOLAR_RADIATION_TBQ02C"
    DEFAULT_MQTT_TOPIC = "rs485/tbq02c"
    MEASUREMENT_KEYS = ("solar_radiation_wm2",)


class XM8504Agent(MQTTSensorAgent):
    SENSOR_NAME = "SONBEST_XM8504"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_RAIN_GAUGE_XM8504"
    DEFAULT_MQTT_TOPIC = "rs485/xm8504"
    MEASUREMENT_KEYS = ("rain_interval_mm",)


class TR4H01XAgent(MQTTSensorAgent):
    SENSOR_NAME = "TR_4H01X"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_SOIL_MOISTURE_TR4H01X"
    DEFAULT_MQTT_TOPIC = "rs485/tr4h01x"
    MEASUREMENT_KEYS = (
        "soil_moisture_probe_1_percent",
        "soil_moisture_probe_2_percent",
        "soil_moisture_probe_3_percent",
        "soil_moisture_probe_4_percent",
    )


class OpticalRainGaugeAgent(MQTTSensorAgent):
    SENSOR_NAME = "OPTICAL_RAIN_GAUGE"
    MQTT_TOPIC_ENV = "MQTT_TOPIC_OPTICAL_RAIN_GAUGE"
    DEFAULT_MQTT_TOPIC = "rs485/optical-rain"
    MEASUREMENT_KEYS = (
        "rainfall_total_mm",
        "rain_interval_mm",
        "rain_intensity_mm_min",
        "illuminance_lux",
    )
