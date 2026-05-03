import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message
from spade.template import Template

from StatusAgent import AgentStatus, StatusAwareAgent, utc_now_iso

load_dotenv()

logger = logging.getLogger(__name__)

MONITORING_ONTOLOGY = "sensor-monitoring"
ALGORITHM_INPUT_ONTOLOGY = "algorithm-input"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def make_sensor_message_template() -> Template:
    template = Template()
    template.set_metadata("ontology", MONITORING_ONTOLOGY)
    template.set_metadata("performative", "inform")
    return template


def expected_sensor_ids() -> list[str]:
    raw = os.getenv(
        "EXPECTED_SENSOR_IDS",
        "sm9560b,thpwnj,tbq02c,sm7005",
    )

    return [item.strip() for item in raw.split(",") if item.strip()]


def compact_sensor_list(items: list[str]) -> str:
    if not items:
        return "none"

    return ",".join(items)


class SensorMessageReceiverBehaviour(CyclicBehaviour):
    async def on_start(self) -> None:
        logger.info("Manager message receiver started")

    async def run(self) -> None:
        msg = await self.receive(timeout=5)

        if msg is None:
            return

        received_at = utc_now_iso()
        now_monotonic = time.monotonic()
        sender_jid = str(msg.sender)
        sender_bare_jid = sender_jid.split("/")[0]

        try:
            body = json.loads(msg.body)
        except json.JSONDecodeError:
            logger.warning("Manager received invalid JSON from %s", sender_bare_jid)
            return

        sensor_id = body.get("sensor_id") or sender_bare_jid.split("@")[0]
        kind = msg.get_metadata("kind") or body.get("kind") or "unknown"

        algorithm_record = None

        async with self.agent.state_lock:
            sensor_state = self.agent.sensor_states.setdefault(
                sensor_id,
                {
                    "sensor_id": sensor_id,
                    "sensor_name": body.get("sensor_name") or sensor_id,
                },
            )

            sensor_state.update(
                {
                    "sensor_id": sensor_id,
                    "sensor_name": body.get("sensor_name")
                    or sensor_state.get("sensor_name")
                    or sensor_id,
                    "source_jid": sender_bare_jid,
                    "last_seen_at": received_at,
                    "_last_seen_monotonic": now_monotonic,
                    "last_message_kind": kind,
                    "reported_status": body.get(
                        "sensor_status",
                        AgentStatus.ONLINE_IDLE.value,
                    ),
                    "reported_message": body.get("sensor_status_message", ""),
                    "last_heartbeat_at": (
                        received_at
                        if kind == "heartbeat"
                        else sensor_state.get("last_heartbeat_at")
                    ),
                }
            )

            if kind == "data":
                sensor_state.update(
                    {
                        "mqtt_topic": body.get("mqtt_topic") or body.get("topic"),
                        "last_data_at": body.get("mqtt_received_at")
                        or body.get("sent_at")
                        or received_at,
                        "_last_data_monotonic": now_monotonic,
                        "payload": body.get("payload"),
                    }
                )

                algorithm_record = self.agent.build_algorithm_record(sensor_state)

                logger.debug("Manager received data from sensor %s", sensor_id)

            elif kind == "heartbeat":
                logger.debug("Manager received heartbeat from sensor %s", sensor_id)

            else:
                logger.debug(
                    "Manager received %s message from sensor %s",
                    kind,
                    sensor_id,
                )

            health = self.agent.compute_health_unlocked()
            self.agent.apply_health_unlocked(health)
            self.agent.write_state_file_unlocked(health)

        await self.send_ack(msg, sensor_id, kind, received_at)

        if algorithm_record is not None:
            await self.forward_to_algorithm_agent(algorithm_record)

    async def send_ack(
        self,
        msg: Message,
        sensor_id: str,
        kind: str,
        received_at: str,
    ) -> None:
        conversation_id = msg.get_metadata("conversation-id")

        if not conversation_id:
            return

        ack = Message(to=str(msg.sender))
        ack.set_metadata("performative", "confirm")
        ack.set_metadata("ontology", MONITORING_ONTOLOGY)
        ack.set_metadata("kind", "ack")
        ack.set_metadata("conversation-id", conversation_id)
        ack.body = json.dumps(
            {
                "kind": "ack",
                "conversation_id": conversation_id,
                "sensor_id": sensor_id,
                "ack_for": kind,
                "received_at": received_at,
                "manager_status": self.agent.current_manager_status,
            },
            ensure_ascii=False,
        )

        await self.send(ack)

    async def forward_to_algorithm_agent(self, record: dict[str, Any]) -> None:
        if not self.agent.algorithm_agent_jid:
            return

        msg = Message(to=self.agent.algorithm_agent_jid)
        msg.set_metadata("performative", "inform")
        msg.set_metadata("ontology", ALGORITHM_INPUT_ONTOLOGY)
        msg.body = json.dumps(record, ensure_ascii=False)

        try:
            await self.send(msg)
            logger.debug(
                "Manager forwarded gateway %s data to algorithm agent",
                record.get("gateway_guid"),
            )
        except Exception:
            logger.exception("Manager failed to forward data to algorithm agent")


class ManagerHealthMonitorBehaviour(PeriodicBehaviour):
    async def on_start(self) -> None:
        logger.info("Manager health monitor started")

    async def run(self) -> None:
        async with self.agent.state_lock:
            health = self.agent.compute_health_unlocked()
            self.agent.apply_health_unlocked(health)
            self.agent.write_state_file_unlocked(health)


class SensorManagerAgent(StatusAwareAgent):
    async def setup(self) -> None:
        logger.info("Manager agent started: %s", self.jid)

        self.state_lock = asyncio.Lock()
        self.expected_sensor_ids = expected_sensor_ids()
        self.started_at_monotonic = time.monotonic()

        self.algorithm_agent_jid = os.getenv(
            "ALGORITHM_AGENT_JID",
            "irrigation@localhost",
        )

        self.sensor_states: dict[str, dict[str, Any]] = {
            sensor_id: {
                "sensor_id": sensor_id,
                "reported_status": AgentStatus.OFFLINE.value,
                "effective_status": AgentStatus.OFFLINE.value,
                "last_seen_at": None,
                "last_data_at": None,
            }
            for sensor_id in self.expected_sensor_ids
        }

        self.current_manager_status = AgentStatus.ONLINE_IDLE.value
        self.current_manager_message = "waiting for sensors"

        self.storage_file = Path(
            os.getenv("SENSOR_STATE_FILE", "./data/sensors_state.json")
        )
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            self.current_manager_message,
            priority=10,
        )
        self.set_offline_message("stopped normally")

        self.add_behaviour(
            SensorMessageReceiverBehaviour(),
            make_sensor_message_template(),
        )

        period = env_float("MANAGER_HEALTH_CHECK_SECONDS", 5.0)
        self.add_behaviour(ManagerHealthMonitorBehaviour(period=period))

    def compute_health_unlocked(self) -> dict[str, Any]:
        now = time.monotonic()

        offline_after = env_float("MANAGER_SENSOR_OFFLINE_AFTER_SECONDS", 30.0)
        idle_after = env_float("SENSOR_IDLE_AFTER_SECONDS", 30.0)
        startup_grace = env_float("MANAGER_STARTUP_GRACE_SECONDS", 15.0)

        working: list[str] = []
        idle: list[str] = []
        degraded: list[str] = []
        offline: list[str] = []
        effective_statuses: dict[str, str] = {}

        for sensor_id in self.expected_sensor_ids:
            sensor_state = self.sensor_states.get(sensor_id, {})

            last_seen_monotonic = sensor_state.get("_last_seen_monotonic")
            last_data_monotonic = sensor_state.get("_last_data_monotonic")
            reported_status = sensor_state.get(
                "reported_status",
                AgentStatus.OFFLINE.value,
            )

            if last_seen_monotonic is None or now - last_seen_monotonic > offline_after:
                offline.append(sensor_id)
                effective_statuses[sensor_id] = AgentStatus.OFFLINE.value
                continue

            if reported_status == AgentStatus.DEGRADED.value:
                degraded.append(sensor_id)
                effective_statuses[sensor_id] = AgentStatus.DEGRADED.value
                continue

            if (
                reported_status == AgentStatus.WORKING.value
                and last_data_monotonic is not None
                and now - last_data_monotonic <= idle_after
            ):
                working.append(sensor_id)
                effective_statuses[sensor_id] = AgentStatus.WORKING.value
                continue

            idle.append(sensor_id)
            effective_statuses[sensor_id] = AgentStatus.ONLINE_IDLE.value

        expected_count = len(self.expected_sensor_ids)
        seen_count = expected_count - len(offline)
        uptime = int(now - self.started_at_monotonic)

        if len(working) == expected_count:
            manager_status = AgentStatus.WORKING
            manager_message = f"all sensors active ({expected_count}/{expected_count})"

        elif seen_count == 0 and uptime < startup_grace:
            manager_status = AgentStatus.ONLINE_IDLE
            manager_message = "waiting for sensors"

        else:
            manager_status = AgentStatus.DEGRADED

            details = [f"active {len(working)}/{expected_count}"]

            if idle:
                details.append(f"idle {compact_sensor_list(idle)}")

            if degraded:
                details.append(f"degraded {compact_sensor_list(degraded)}")

            if offline:
                details.append(f"offline {compact_sensor_list(offline)}")

            manager_message = "; ".join(details)

        return {
            "manager_status": manager_status,
            "manager_message": manager_message,
            "working": working,
            "idle": idle,
            "degraded": degraded,
            "offline": offline,
            "effective_statuses": effective_statuses,
            "checked_at": utc_now_iso(),
        }

    def apply_health_unlocked(self, health: dict[str, Any]) -> None:
        manager_status: AgentStatus = health["manager_status"]
        manager_message: str = health["manager_message"]

        previous_status = self.current_manager_status
        previous_message = self.current_manager_message

        for sensor_id, effective_status in health["effective_statuses"].items():
            self.sensor_states.setdefault(sensor_id, {})[
                "effective_status"
            ] = effective_status

        self.current_manager_status = manager_status.value
        self.current_manager_message = manager_message

        self.set_agent_status(
            manager_status,
            manager_message,
            priority=10,
        )
        self.set_offline_message(
            f"stopped normally; previous status {manager_status.value}"
        )

        if previous_status != manager_status.value or previous_message != manager_message:
            if manager_status == AgentStatus.DEGRADED:
                logger.warning(
                    "Manager status changed to %s: %s",
                    manager_status.value,
                    manager_message,
                )
            else:
                logger.info(
                    "Manager status changed to %s: %s",
                    manager_status.value,
                    manager_message,
                )

    def extract_gateway_guid(self, sensor_state: dict[str, Any]) -> str | None:
        mqtt_topic = sensor_state.get("mqtt_topic")

        if not mqtt_topic:
            return None

        return str(mqtt_topic).split("/")[-1]

    def extract_timestamp_and_value(
        self,
        sensor_state: dict[str, Any],
    ) -> tuple[str | None, Any]:
        payload = sensor_state.get("payload")

        if isinstance(payload, dict):
            timestamp = (
                payload.get("timestamp")
                or sensor_state.get("last_data_at")
                or sensor_state.get("last_seen_at")
            )

            if "value" in payload and len(payload) <= 2:
                value = payload["value"]
            else:
                value = {
                    key: item
                    for key, item in payload.items()
                    if key != "timestamp"
                }

            return timestamp, value

        timestamp = (
            sensor_state.get("last_data_at")
            or sensor_state.get("last_seen_at")
        )

        return timestamp, payload

    def build_algorithm_record(
        self,
        sensor_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        gateway_guid = self.extract_gateway_guid(sensor_state)
        timestamp, value = self.extract_timestamp_and_value(sensor_state)

        if not gateway_guid or timestamp is None or value is None:
            return None

        return {
            "gateway_guid": gateway_guid,
            "timestamp": timestamp,
            "value": value,
        }

    def write_state_file_unlocked(self, health: dict[str, Any]) -> None:
        records = []

        for sensor_id in self.expected_sensor_ids:
            sensor_state = self.sensor_states.get(sensor_id, {})
            record = self.build_algorithm_record(sensor_state)

            if record is not None:
                records.append(record)

        tmp_file = self.storage_file.with_name(f".{self.storage_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.storage_file)