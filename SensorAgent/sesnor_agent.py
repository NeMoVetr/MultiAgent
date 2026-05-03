import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from aiomqtt import Client, MqttError
from dotenv import load_dotenv
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template

from StatusAgent import AgentStatus, StatusAwareAgent, utc_now_iso

load_dotenv()

logger = logging.getLogger(__name__)

MONITORING_ONTOLOGY = "sensor-monitoring"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def make_ack_template() -> Template:
    template = Template()
    template.set_metadata("ontology", MONITORING_ONTOLOGY)
    template.set_metadata("performative", "confirm")
    return template


class SensorAgentBase(StatusAwareAgent):
    async def setup_sensor_agent(
        self,
        sensor_id: str,
        sensor_name: str,
        mqtt_topic: str,
    ) -> None:
        self.sensor_id = sensor_id
        self.sensor_name = sensor_name
        self.mqtt_topic = mqtt_topic
        self.manager_jid = self.get("manager_jid") or os.getenv(
            "MANAGER_JID",
            "manager@localhost",
        )

        self.status_lock = asyncio.Lock()
        self.pending_acks: dict[str, asyncio.Future] = {}

        self.mqtt_subscribed = False
        self.mqtt_error: str | None = None

        self.last_mqtt_data_at: str | None = None
        self.last_mqtt_data_monotonic: float | None = None

        self.last_manager_ack_at: str | None = None
        self.last_manager_ack_monotonic: float | None = None
        self.manager_available: bool | None = None

        self.current_operational_status = AgentStatus.ONLINE_IDLE.value
        self.current_status_message = f"{self.sensor_name} starting"

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            self.current_status_message,
            priority=5,
        )
        self.set_offline_message("stopped normally")

        self.add_behaviour(ManagerAckReceiverBehaviour(), make_ack_template())
        self.add_behaviour(MQTTListenerBehaviour())

        heartbeat_period = env_float("SENSOR_HEARTBEAT_SECONDS", 10.0)
        self.add_behaviour(SensorHealthHeartbeatBehaviour(period=heartbeat_period))

        logger.info(
            "Sensor agent configured: %s, topic=%s, manager=%s",
            self.sensor_id,
            self.mqtt_topic,
            self.manager_jid,
        )

    def register_pending_ack(self, conversation_id: str) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self.pending_acks[conversation_id] = future
        return future

    def remove_pending_ack(self, conversation_id: str) -> None:
        self.pending_acks.pop(conversation_id, None)

    def resolve_pending_ack(self, conversation_id: str, msg: Message) -> None:
        future = self.pending_acks.pop(conversation_id, None)

        if future and not future.done():
            future.set_result(msg)

    async def set_manager_available(self, value: bool) -> None:
        async with self.status_lock:
            self.manager_available = value

            if value:
                now = utc_now_iso()
                self.last_manager_ack_at = now
                self.last_manager_ack_monotonic = time.monotonic()

        await self.refresh_presence_status()

    async def set_mqtt_subscribed(self, value: bool, error: str | None = None) -> None:
        async with self.status_lock:
            self.mqtt_subscribed = value
            self.mqtt_error = error

        await self.refresh_presence_status()

    async def record_mqtt_data(self, payload: Any) -> None:
        payload_timestamp = None

        if isinstance(payload, dict):
            payload_timestamp = payload.get("timestamp")

        async with self.status_lock:
            self.last_mqtt_data_at = payload_timestamp or utc_now_iso()
            self.last_mqtt_data_monotonic = time.monotonic()
            self.mqtt_error = None
            self.mqtt_subscribed = True

        await self.refresh_presence_status()

    def _mqtt_data_age_unlocked(self) -> int | None:
        if self.last_mqtt_data_monotonic is None:
            return None

        return int(time.monotonic() - self.last_mqtt_data_monotonic)

    async def refresh_presence_status(self) -> None:
        idle_after = env_float("SENSOR_IDLE_AFTER_SECONDS", 30.0)

        async with self.status_lock:
            previous_status = self.current_operational_status
            previous_message = self.current_status_message

            mqtt_age = self._mqtt_data_age_unlocked()

            if self.mqtt_error:
                status = AgentStatus.DEGRADED
                message = f"{self.sensor_name} MQTT error"

            elif self.manager_available is False:
                status = AgentStatus.DEGRADED
                message = f"{self.sensor_name} manager unavailable"

            elif not self.mqtt_subscribed:
                status = AgentStatus.ONLINE_IDLE
                message = f"{self.sensor_name} connecting to MQTT"

            elif self.last_mqtt_data_at is None:
                status = AgentStatus.ONLINE_IDLE
                message = f"{self.sensor_name} waiting for MQTT data"

            elif mqtt_age is not None and mqtt_age > idle_after:
                status = AgentStatus.ONLINE_IDLE
                message = f"{self.sensor_name} no MQTT data for {mqtt_age}s"

            else:
                status = AgentStatus.WORKING
                message = f"{self.sensor_name} receiving MQTT data"

            self.current_operational_status = status.value
            self.current_status_message = message
            self.set_offline_message(f"stopped normally; previous status {status.value}")

        self.set_agent_status(status, message, priority=5)

        if previous_status != status.value or previous_message != message:
            if status == AgentStatus.DEGRADED:
                logger.warning(
                    "Sensor %s status changed to %s: %s",
                    self.sensor_id,
                    status.value,
                    message,
                )
            else:
                logger.info(
                    "Sensor %s status changed to %s: %s",
                    self.sensor_id,
                    status.value,
                    message,
                )


class ManagerMessageMixin:
    async def send_to_manager_with_ack(
        self,
        kind: str,
        extra_body: dict[str, Any],
        timeout: float | None = None,
    ) -> bool:
        timeout = timeout or env_float("MANAGER_ACK_TIMEOUT_SECONDS", 3.0)

        conversation_id = uuid.uuid4().hex
        future = self.agent.register_pending_ack(conversation_id)

        body = {
            "kind": kind,
            "sensor_id": self.agent.sensor_id,
            "sensor_name": self.agent.sensor_name,
            "topic": self.agent.mqtt_topic,
            "sensor_status": self.agent.current_operational_status,
            "sensor_status_message": self.agent.current_status_message,
            "sent_at": utc_now_iso(),
            **extra_body,
        }

        msg = Message(to=self.agent.manager_jid)
        msg.set_metadata("performative", "inform")
        msg.set_metadata("ontology", MONITORING_ONTOLOGY)
        msg.set_metadata("kind", kind)
        msg.set_metadata("conversation-id", conversation_id)
        msg.body = json.dumps(body, ensure_ascii=False)

        try:
            await self.send(msg)

        except Exception:
            self.agent.remove_pending_ack(conversation_id)
            await self.agent.set_manager_available(False)

            logger.exception(
                "Sensor %s failed to send %s message to manager",
                self.agent.sensor_id,
                kind,
            )

            return False

        try:
            await asyncio.wait_for(future, timeout=timeout)
            await self.agent.set_manager_available(True)
            return True

        except asyncio.TimeoutError:
            self.agent.remove_pending_ack(conversation_id)
            await self.agent.set_manager_available(False)

            logger.warning(
                "Sensor %s did not receive manager ACK for %s within %.1fs",
                self.agent.sensor_id,
                kind,
                timeout,
            )

            return False


class ManagerAckReceiverBehaviour(CyclicBehaviour):
    async def run(self) -> None:
        msg = await self.receive(timeout=5)

        if msg is None:
            return

        conversation_id = msg.get_metadata("conversation-id")

        if not conversation_id and msg.body:
            try:
                body = json.loads(msg.body)
                conversation_id = body.get("conversation_id")
            except json.JSONDecodeError:
                logger.warning("Received invalid ACK body from manager")
                return

        if conversation_id:
            self.agent.resolve_pending_ack(conversation_id, msg)


class SensorHealthHeartbeatBehaviour(ManagerMessageMixin, PeriodicBehaviour):
    async def on_start(self) -> None:
        logger.info(
            "Sensor %s heartbeat started",
            self.agent.sensor_id,
        )

    async def run(self) -> None:
        await self.agent.refresh_presence_status()

        await self.send_to_manager_with_ack(
            kind="heartbeat",
            extra_body={
                "last_mqtt_data_at": self.agent.last_mqtt_data_at,
                "last_manager_ack_at": self.agent.last_manager_ack_at,
            },
        )

        await self.agent.refresh_presence_status()


class MQTTListenerBehaviour(ManagerMessageMixin, CyclicBehaviour):
    async def on_start(self) -> None:
        logger.info(
            "Sensor %s MQTT listener started",
            self.agent.sensor_id,
        )

    async def run(self) -> None:
        mqtt_host = os.getenv("MQTT_HOST", "localhost")
        mqtt_port = int(os.getenv("MQTT_PORT", "1883"))

        try:
            async with Client(mqtt_host, mqtt_port) as client:
                await client.subscribe(self.agent.mqtt_topic)
                await self.agent.set_mqtt_subscribed(True)

                logger.info(
                    "Sensor %s subscribed to MQTT topic %s",
                    self.agent.sensor_id,
                    self.agent.mqtt_topic,
                )

                async for message in client.messages:
                    raw_payload = message.payload.decode(errors="replace")

                    try:
                        payload = json.loads(raw_payload)
                    except json.JSONDecodeError:
                        payload = {"raw": raw_payload}
                        logger.warning(
                            "Sensor %s received non-JSON MQTT payload",
                            self.agent.sensor_id,
                        )

                    await self.agent.record_mqtt_data(payload)

                    logger.debug(
                        "Sensor %s received MQTT message from %s: %s",
                        self.agent.sensor_id,
                        message.topic,
                        payload,
                    )

                    ack_ok = await self.send_to_manager_with_ack(
                        kind="data",
                        extra_body={
                            "mqtt_topic": str(message.topic),
                            "mqtt_received_at": utc_now_iso(),
                            "payload": payload,
                        },
                    )

                    if ack_ok:
                        logger.debug(
                            "Sensor %s received manager ACK for data message",
                            self.agent.sensor_id,
                        )

                    await self.agent.refresh_presence_status()

        except MqttError as error:
            await self.agent.set_mqtt_subscribed(False, error=str(error))

            logger.warning(
                "Sensor %s MQTT connection lost: %s",
                self.agent.sensor_id,
                error,
            )

            await asyncio.sleep(5)

        except Exception as error:
            await self.agent.set_mqtt_subscribed(False, error=str(error))

            logger.exception(
                "Sensor %s critical MQTT listener error",
                self.agent.sensor_id,
            )

            await asyncio.sleep(5)


class SM9560BAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="sm9560b",
            sensor_name="SONBEST SM9560B",
            mqtt_topic="sensors/8f3e2a1c-9b7d-4e5f-a1b2-c3d4e5f6a7b8",
        )


class THPWNJAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="thpwnj",
            sensor_name="Veinasa THPW-NJ",
            mqtt_topic="sensors/1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
        )


class TBQ02CAgent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="tbq02c",
            sensor_name="XS-TBQ02C",
            mqtt_topic="sensors/9f8e7d6c-5b4a-3f2e-1d0c-b9a8f7e6d5c4",
        )


class SM7005Agent(SensorAgentBase):
    async def setup(self) -> None:
        await self.setup_sensor_agent(
            sensor_id="sm7005",
            sensor_name="SONBEST SM7005",
            mqtt_topic="sensors/4b5c6d7e-8f9a-0b1c-2d3e-4f5a6b7c8d9e",
        )