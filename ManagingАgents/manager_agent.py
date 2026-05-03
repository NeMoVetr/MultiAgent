# manager_agent.py

import asyncio
import json
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

MONITORING_ONTOLOGY = "sensor-monitoring"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def make_internal_template(name: str) -> Template:
    template = Template()
    template.set_metadata("__internal_behaviour__", name)
    return template


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


class SensorMessageReceiverBehaviour(CyclicBehaviour):
    async def on_start(self) -> None:
        print("[MANAGER] Sensor message receiver started")

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
            print(f"[MANAGER] Invalid JSON from {sender_jid}: {msg.body}")
            return

        sensor_id = body.get("sensor_id") or sender_bare_jid.split("@")[0]
        kind = msg.get_metadata("kind") or body.get("kind") or "unknown"

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
                    "reported_status_detail": body.get("sensor_status_detail", ""),
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

            health = self.agent.compute_health_unlocked()
            self.agent.apply_health_unlocked(health)
            self.agent.write_state_file_unlocked(health)

        await self.send_ack(msg, sensor_id, kind, received_at)

        print(
            f"[MANAGER] Received {kind} from {sensor_id}; "
            f"manager_status={self.agent.current_manager_status}"
        )

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


class ManagerHealthMonitorBehaviour(PeriodicBehaviour):
    async def on_start(self) -> None:
        print("[MANAGER] Health monitor started")

    async def run(self) -> None:
        async with self.agent.state_lock:
            health = self.agent.compute_health_unlocked()
            self.agent.apply_health_unlocked(health)
            self.agent.write_state_file_unlocked(health)


class SensorManagerAgent(StatusAwareAgent):
    async def setup(self) -> None:
        print(f"[MANAGER] Agent started: {self.jid}")

        self.state_lock = asyncio.Lock()
        self.expected_sensor_ids = expected_sensor_ids()
        self.started_at_monotonic = time.monotonic()

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
        self.current_manager_detail = (
            f"role=manager | waiting_for_sensors | expected={len(self.expected_sensor_ids)}"
        )

        self.storage_file = Path(
            os.getenv("SENSOR_STATE_FILE", "./data/sensors_state.json")
        )
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            self.current_manager_detail,
            priority=10,
        )

        self.set_offline_detail(
            f"role=manager | last_status={AgentStatus.ONLINE_IDLE.value} | stopped_at={utc_now_iso()}"
        )

        self.add_behaviour(
            SensorMessageReceiverBehaviour(),
            make_sensor_message_template(),
        )

        period = env_float("MANAGER_HEALTH_CHECK_SECONDS", 5.0)

        self.add_behaviour(
            ManagerHealthMonitorBehaviour(period=period),
            make_internal_template("manager-health-monitor"),
        )

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
            detail = (
                f"role=manager | sensors_working={len(working)}/{expected_count} "
                f"| idle=none | degraded=none | offline=none"
            )

        elif seen_count == 0 and uptime < startup_grace:
            manager_status = AgentStatus.ONLINE_IDLE
            detail = (
                f"role=manager | waiting_for_sensors | seen=0/{expected_count} "
                f"| grace={startup_grace}s"
            )

        else:
            manager_status = AgentStatus.DEGRADED
            detail = (
                f"role=manager | sensors_working={len(working)}/{expected_count} "
                f"| idle={','.join(idle) or 'none'} "
                f"| degraded={','.join(degraded) or 'none'} "
                f"| offline={','.join(offline) or 'none'}"
            )

        return {
            "manager_status": manager_status,
            "manager_detail": detail,
            "working": working,
            "idle": idle,
            "degraded": degraded,
            "offline": offline,
            "effective_statuses": effective_statuses,
            "checked_at": utc_now_iso(),
        }

    def apply_health_unlocked(self, health: dict[str, Any]) -> None:
        manager_status: AgentStatus = health["manager_status"]
        manager_detail: str = health["manager_detail"]

        for sensor_id, effective_status in health["effective_statuses"].items():
            self.sensor_states.setdefault(sensor_id, {})[
                "effective_status"
            ] = effective_status

        self.current_manager_status = manager_status.value
        self.current_manager_detail = manager_detail

        self.set_agent_status(
            manager_status,
            manager_detail,
            priority=10,
        )

        self.set_offline_detail(
            f"role=manager | last_status={manager_status.value} | {manager_detail}"
        )

    def write_state_file_unlocked(self, health: dict[str, Any]) -> None:
        state = {
            "updated_at": health["checked_at"],
            "manager": {
                "jid": str(self.jid).split("/")[0],
                "status": health["manager_status"].value,
                "detail": health["manager_detail"],
                "working_sensors": health["working"],
                "idle_sensors": health["idle"],
                "degraded_sensors": health["degraded"],
                "offline_sensors": health["offline"],
            },
            "sensors": {},
        }

        for sensor_id in self.expected_sensor_ids:
            source = self.sensor_states.get(sensor_id, {})

            clean = {
                key: value
                for key, value in source.items()
                if not key.startswith("_")
            }

            clean.setdefault("sensor_id", sensor_id)
            clean.setdefault(
                "effective_status",
                health["effective_statuses"].get(
                    sensor_id,
                    AgentStatus.OFFLINE.value,
                ),
            )

            state["sensors"][sensor_id] = clean

        tmp_file = self.storage_file.with_name(f".{self.storage_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.storage_file)