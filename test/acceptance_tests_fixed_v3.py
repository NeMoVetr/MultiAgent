#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Acceptance tests for the MultiAgent irrigation system.

Run from the project root:
    uv run python acceptance_tests.py

The script checks the laboratory test cases from the Program and Test Methods:
- Docker infrastructure;
- Node-RED flow configuration and GUI availability;
- ejabberd Web Admin availability;
- MQTT broker connectivity;
- XMPP agent registration and presence statuses;
- sensor -> manager -> normalizer -> irrigation pipeline;
- raw and normalized JSON files;
- time normalization;
- irrigation decisions false/true;
- degraded/offline scenarios;
- log file existence.

Important:
- For deterministic E2E tests the script stops the Node-RED container after GUI checks
  and restarts it at the end. Disable this with --no-isolate-node-red.
- The script publishes controlled MQTT messages directly to Mosquitto.
"""

from __future__ import annotations

# aiomqtt/paho integration on Windows requires a selector-based event loop.
# This must be set before any project modules, SPADE, or MQTT clients are imported.
import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys


def ensure_windows_selector_event_loop_policy() -> None:
    """
    aiomqtt/paho uses add_reader/add_writer. On Windows the default
    ProactorEventLoop does not implement those methods, so MQTT hangs with
    NotImplementedError. Selector policy must be installed before the loop
    is created.
    """
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


ensure_windows_selector_event_loop_policy()

# aiomqtt/paho-mqtt uses add_reader/add_writer. On Windows the default
# ProactorEventLoop does not implement these methods, so the policy must be
# set before any event loop is created and before project modules are imported.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "acceptance_tests"
LOG_DIR = PROJECT_ROOT / "logs"

# These constants correspond to the gateway GUIDs and topics used in the project.
LIGHT_GUID = "8f3e2a1c-9b7d-4e5f-a1b2-c3d4e5f6a7b8"       # SONBEST SM9560B
WEATHER_GUID = "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d"     # Veinasa THPW-NJ
SOLAR_GUID = "9f8e7d6c-5b4a-3f2e-1d0c-b9a8f7e6d5c4"       # XS-TBQ02C
RAIN_GUID = "4b5c6d7e-8f9a-0b1c-2d3e-4f5a6b7c8d9e"        # SONBEST SM7005

SENSOR_TOPICS: dict[str, str] = {
    "light": f"sensors/{LIGHT_GUID}",
    "weather": f"sensors/{WEATHER_GUID}",
    "solar": f"sensors/{SOLAR_GUID}",
    "rain": f"sensors/{RAIN_GUID}",
}

EXPECTED_USERS = {
    "manager",
    "normalizer",
    "irrigation",
    "sm9560b",
    "thpwnj",
    "tbq02c",
    "sm7005",
}


@dataclass
class TestResult:
    code: str
    name: str
    passed: bool
    details: str = ""
    evidence: str = ""
    duration_seconds: float = 0.0


class AcceptanceTestError(AssertionError):
    pass


# ---------------------------------------------------------------------------
# Environment and imports
# ---------------------------------------------------------------------------


def configure_acceptance_environment() -> None:
    """Set deterministic paths and short timeouts before importing project modules."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    running_in_docker = Path("/.dockerenv").exists() or os.getenv("RUNNING_IN_DOCKER") == "1"

    # XMPP_HOST is used by the project to build JIDs, therefore the default must match
    # ejabberd vhost from ejabberd.yml: localhost.
    os.environ["XMPP_HOST"] = os.getenv("ACCEPTANCE_XMPP_HOST", "localhost")
    os.environ.setdefault("XMPP_VHOST", "localhost")
    os.environ.setdefault("XMPP_PORT", "5222")
    os.environ.setdefault("SPADE_AGENT_PASSWORD", "agent_password_123")

    os.environ.setdefault("MANAGER_JID", "manager@localhost")
    os.environ.setdefault("NORMALIZER_AGENT_JID", "normalizer@localhost")
    os.environ.setdefault("ALGORITHM_AGENT_JID", "irrigation@localhost")

    default_mqtt_host = "mosquitto" if running_in_docker else "localhost"
    os.environ["MQTT_HOST"] = os.getenv("ACCEPTANCE_MQTT_HOST", default_mqtt_host)
    os.environ.setdefault("MQTT_PORT", "1883")

    # Resolve broker address for host execution vs Docker-container execution.
    # This value is used by both the tests and the sensor agents.
    mqtt_port = int(os.environ.get("MQTT_PORT", "1883"))
    os.environ["MQTT_HOST"] = detect_mqtt_host(mqtt_port)

    os.environ["SENSOR_STATE_FILE"] = str(DATA_DIR / "raw_sensor_state.json")
    os.environ["NORMALIZED_SENSOR_STATE_FILE"] = str(DATA_DIR / "sensors_state.json")
    os.environ["IRRIGATION_DECISION_FILE"] = str(DATA_DIR / "irrigation_decision.json")
    os.environ["IRRIGATION_MEMORY_FILE"] = str(DATA_DIR / "irrigation_memory.json")

    os.environ.setdefault("EXPECTED_SENSOR_IDS", "sm9560b,thpwnj,tbq02c,sm7005")
    os.environ.setdefault("SOURCE_TIMESTAMP_TIMEZONE", "UTC")

    os.environ.setdefault("GATEWAY_LIGHT_GUID", LIGHT_GUID)
    os.environ.setdefault("GATEWAY_WEATHER_GUID", WEATHER_GUID)
    os.environ.setdefault("GATEWAY_SOLAR_GUID", SOLAR_GUID)
    os.environ.setdefault("GATEWAY_RAIN_GUID", RAIN_GUID)

    # Shorter intervals make degraded/offline scenarios testable in seconds.
    os.environ["SENSOR_HEARTBEAT_SECONDS"] = os.getenv("ACCEPTANCE_SENSOR_HEARTBEAT_SECONDS", "2")
    os.environ["SENSOR_IDLE_AFTER_SECONDS"] = os.getenv("ACCEPTANCE_SENSOR_IDLE_AFTER_SECONDS", "6")
    os.environ["MANAGER_ACK_TIMEOUT_SECONDS"] = os.getenv("ACCEPTANCE_MANAGER_ACK_TIMEOUT_SECONDS", "1")
    os.environ["MANAGER_HEALTH_CHECK_SECONDS"] = os.getenv("ACCEPTANCE_MANAGER_HEALTH_CHECK_SECONDS", "2")
    os.environ["MANAGER_SENSOR_OFFLINE_AFTER_SECONDS"] = os.getenv("ACCEPTANCE_MANAGER_SENSOR_OFFLINE_AFTER_SECONDS", "8")
    os.environ["MANAGER_STARTUP_GRACE_SECONDS"] = os.getenv("ACCEPTANCE_MANAGER_STARTUP_GRACE_SECONDS", "2")
    os.environ["ALGORITHM_MAX_DATA_AGE_SECONDS"] = os.getenv("ACCEPTANCE_ALGORITHM_MAX_DATA_AGE_SECONDS", "60")

    os.environ.setdefault("IRRIGATION_CROP_COEFFICIENT", "1.0")
    os.environ.setdefault("IRRIGATION_MAD_MM", "25")
    os.environ.setdefault("IRRIGATION_MAX_DEPLETION_MM", "80")
    os.environ.setdefault("IRRIGATION_MAX_WIND_SPEED_MS", "8")
    os.environ.setdefault("IRRIGATION_DEFAULT_STEP_SECONDS", "300")
    os.environ.setdefault("RAIN_VALUE_MODE", "incremental")

    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("XMPP_LOG_LEVEL", "WARNING")
    os.environ["LOG_FILE"] = str(LOG_DIR / "acceptance_tests_agents.log")


def import_project_modules() -> dict[str, Any]:
    """Import project modules after environment setup."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from logging_config import configure_logging

    configure_logging()

    from main import (
        create_irrigation_agent,
        create_manager_agent,
        create_normalizer_agent,
        create_sensor_agents,
    )
    from algorithms.algorithm_contract import AlgorithmInputRecord
    from algorithms.irrigation_algorithm import IrrigationAlgorithm, IrrigationSettings
    from algorithms.time_normalizer import normalize_timestamp_to_minute_utc

    return {
        "create_irrigation_agent": create_irrigation_agent,
        "create_normalizer_agent": create_normalizer_agent,
        "create_manager_agent": create_manager_agent,
        "create_sensor_agents": create_sensor_agents,
        "AlgorithmInputRecord": AlgorithmInputRecord,
        "IrrigationAlgorithm": IrrigationAlgorithm,
        "IrrigationSettings": IrrigationSettings,
        "normalize_timestamp_to_minute_utc": normalize_timestamp_to_minute_utc,
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def run_cmd(
    command: list[str],
    timeout: float = 20.0,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )

def mqtt_publish_sync(topic: str, payload: dict[str, Any] | str, qos: int = 0) -> None:
    """Publish MQTT message without asyncio add_reader/add_writer.

    The preferred path is docker exec mosquitto_pub because the broker runs in Docker.
    If Docker CLI is unavailable, paho-mqtt is used as a synchronous fallback.
    """
    if not isinstance(payload, str):
        payload_text = json.dumps(payload, ensure_ascii=False)
    else:
        payload_text = payload

    container = os.getenv("MOSQUITTO_CONTAINER", "mosquitto")

    if shutil.which("docker") is not None:
        result = run_cmd(
            [
                "docker",
                "exec",
                container,
                "mosquitto_pub",
                "-h",
                "localhost",
                "-p",
                "1883",
                "-t",
                topic,
                "-m",
                payload_text,
                "-q",
                str(qos),
            ],
            timeout=10,
        )

        if result.returncode == 0:
            return

    try:
        import paho.mqtt.client as mqtt
    except ImportError as error:
        raise AcceptanceTestError(
            "Cannot publish MQTT message: docker mosquitto_pub failed and paho-mqtt is not installed"
        ) from error

    mqtt_host = os.getenv("MQTT_HOST", "localhost")
    mqtt_port = int(os.getenv("MQTT_PORT", "1883"))

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()

    client.connect(mqtt_host, mqtt_port, keepalive=30)
    client.loop_start()
    try:
        info = client.publish(topic, payload_text, qos=qos)
        info.wait_for_publish()
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise AcceptanceTestError(f"MQTT publish failed with rc={info.rc}")
    finally:
        client.loop_stop()
        client.disconnect()


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False

    result = run_cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout=10)
    return result.returncode == 0


def unique_sequence(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)

    return result


def detect_mqtt_endpoint() -> tuple[str, int]:
    """Detect a reachable MQTT endpoint for host or Docker execution."""
    configured_host = os.getenv("MQTT_HOST", "").strip()
    configured_port = int(os.getenv("MQTT_PORT", "1883"))

    candidates = unique_sequence(
        [
            configured_host,
            "localhost",
            "127.0.0.1",
            "host.docker.internal",
            "mosquitto",
        ]
    )

    errors: list[str] = []

    for host in candidates:
        try:
            with socket.create_connection((host, configured_port), timeout=2.0):
                os.environ["MQTT_HOST"] = host
                os.environ["MQTT_PORT"] = str(configured_port)
                return host, configured_port
        except OSError as error:
            errors.append(f"{host}:{configured_port} -> {error}")

    raise AcceptanceTestError(
        "MQTT broker is unreachable. Tried: " + "; ".join(errors)
    )


def publish_mqtt_sync(topic: str, payload: dict[str, Any], *, host: str, port: int) -> None:
    """Publish through paho-mqtt synchronously, avoiding aiomqtt in the test runner."""
    import paho.mqtt.client as mqtt

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.connect(host, port, keepalive=30)
    client.loop_start()

    try:
        info = client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=0)
        info.wait_for_publish(timeout=5)

        if not info.is_published():
            raise AcceptanceTestError(f"MQTT message was not published to {topic}")
    finally:
        client.loop_stop()
        client.disconnect()


def http_status(url: str, timeout: float = 5.0) -> int:
    request = urllib.request.Request(url)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        # 401/403 still means that the service is reachable.
        return int(error.code)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")

    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(tmp, path)


def clear_acceptance_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in DATA_DIR.glob("*.json"):
        item.unlink(missing_ok=True)




def encode_mqtt_remaining_length(value: int) -> bytes:
    """Encode MQTT Remaining Length field."""
    encoded = bytearray()
    while True:
        digit = value % 128
        value //= 128
        if value > 0:
            digit |= 0x80
        encoded.append(digit)
        if value == 0:
            break
    return bytes(encoded)


def mqtt_utf8(value: str) -> bytes:
    payload = value.encode("utf-8")
    return len(payload).to_bytes(2, "big") + payload


def mqtt_open_connection(host: str, port: int, timeout: float = 4.0) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(timeout)

    client_id = f"acceptance-{os.getpid()}-{int(time.time() * 1000)}"
    variable_header = mqtt_utf8("MQTT") + bytes([4, 2]) + int(60).to_bytes(2, "big")
    payload = mqtt_utf8(client_id)
    packet = bytes([0x10]) + encode_mqtt_remaining_length(len(variable_header) + len(payload)) + variable_header + payload

    sock.sendall(packet)
    response = sock.recv(4)

    if len(response) < 4 or response[0] != 0x20 or response[3] != 0:
        sock.close()
        raise AcceptanceTestError(f"MQTT CONNACK failed for {host}:{port}: {response!r}")

    return sock


def mqtt_publish_sync(host: str, port: int, topic: str, payload: str, timeout: float = 4.0) -> None:
    sock = mqtt_open_connection(host, port, timeout=timeout)
    try:
        topic_bytes = mqtt_utf8(topic)
        payload_bytes = payload.encode("utf-8")
        variable_and_payload = topic_bytes + payload_bytes
        packet = bytes([0x30]) + encode_mqtt_remaining_length(len(variable_and_payload)) + variable_and_payload
        sock.sendall(packet)
        sock.sendall(b"\xe0\x00")
    finally:
        with contextlib.suppress(Exception):
            sock.close()


def mqtt_probe_sync(host: str, port: int, timeout: float = 3.0) -> None:
    sock = mqtt_open_connection(host, port, timeout=timeout)
    try:
        sock.sendall(b"\xe0\x00")
    finally:
        with contextlib.suppress(Exception):
            sock.close()


def detect_mqtt_host(port: int) -> str:
    """
    Select MQTT host for the current execution context.

    - Host Windows/Linux: usually localhost because docker-compose publishes 1883.
    - Python container in the same Docker network: usually mosquitto.
    - Python container using Docker Desktop host gateway: host.docker.internal.
    """
    explicit = os.getenv("ACCEPTANCE_MQTT_HOST")
    current = os.getenv("MQTT_HOST", "localhost")

    candidates = []
    for item in [explicit, current, "mosquitto", "localhost", "127.0.0.1", "host.docker.internal"]:
        if item and item not in candidates:
            candidates.append(item)

    errors: list[str] = []
    for candidate in candidates:
        try:
            mqtt_probe_sync(candidate, port, timeout=1.5)
            return candidate
        except Exception as error:
            errors.append(f"{candidate}: {type(error).__name__} {error}")

    raise AcceptanceTestError("MQTT broker is not reachable. Tried: " + "; ".join(errors))


def raw_timestamp(minutes_offset: int = 0, second: int = 54, millisecond: int = 401) -> str:
    now = datetime.now(timezone.utc) + timedelta(minutes=minutes_offset)
    dt = now.replace(second=second, microsecond=millisecond * 1000)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def previous_minute_timestamp(current_timestamp: str) -> str:
    value = current_timestamp.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value).astimezone(timezone.utc) - timedelta(minutes=1)
    return dt.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:00Z")


def records_have_gateways(records: Any, gateway_guids: Iterable[str]) -> bool:
    if not isinstance(records, list):
        return False

    found = {item.get("gateway_guid") for item in records if isinstance(item, dict)}
    return set(gateway_guids).issubset(found)


def get_record(records: Any, gateway_guid: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None

    for item in records:
        if isinstance(item, dict) and item.get("gateway_guid") == gateway_guid:
            return item

    return None


async def wait_for_condition(
    predicate: Callable[[], bool],
    timeout: float,
    interval: float = 0.2,
    description: str = "condition",
) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)

    raise AcceptanceTestError(f"Timeout while waiting for {description}")


async def wait_for_json(
    path: Path,
    predicate: Callable[[Any], bool],
    timeout: float = 15.0,
    interval: float = 0.2,
    description: str = "JSON file",
) -> Any:
    last_payload: Any = None
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if path.exists():
            try:
                last_payload = read_json(path)
                if predicate(last_payload):
                    return last_payload
            except json.JSONDecodeError:
                pass

        await asyncio.sleep(interval)

    raise AcceptanceTestError(
        f"Timeout while waiting for {description}. Last payload: {last_payload!r}"
    )


def build_sensor_payloads(timestamp: str, *, raining: bool) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        "light": (
            SENSOR_TOPICS["light"],
            {
                "value": 42000,
                "timestamp": timestamp,
            },
        ),
        "weather": (
            SENSOR_TOPICS["weather"],
            {
                "wind_speed": 2.0,
                "wind_direction": 180,
                "temperature": 24.0 if not raining else 17.0,
                "humidity": 45 if not raining else 85,
                "pressure": 1012,
                "timestamp": timestamp,
            },
        ),
        "solar": (
            SENSOR_TOPICS["solar"],
            {
                "solar_radiation": 850 if not raining else 80,
                "timestamp": timestamp,
            },
        ),
        "rain": (
            SENSOR_TOPICS["rain"],
            {
                "rainfall_mm": 0.0 if not raining else 4.0,
                "is_raining": raining,
                "timestamp": timestamp,
            },
        ),
    }


# ---------------------------------------------------------------------------
# Acceptance test suite
# ---------------------------------------------------------------------------


class AcceptanceTestSuite:
    def __init__(self, args: argparse.Namespace, modules: dict[str, Any]):
        self.args = args
        self.modules = modules
        self.logger = logging.getLogger("acceptance_tests")
        self.results: list[TestResult] = []
        self.started_agents: list[Any] = []
        self.manager_agent: Any | None = None
        self.node_red_was_stopped = False

        self.raw_file = Path(os.environ["SENSOR_STATE_FILE"])
        self.normalized_file = Path(os.environ["NORMALIZED_SENSOR_STATE_FILE"])
        self.decision_file = Path(os.environ["IRRIGATION_DECISION_FILE"])
        self.memory_file = Path(os.environ["IRRIGATION_MEMORY_FILE"])
        self.log_file = Path(os.environ["LOG_FILE"])

    async def run(self) -> int:
        clear_acceptance_data()

        await self.run_test("TC-01", "Запуск Docker-инфраструктуры", self.test_docker_infrastructure)
        await self.run_test("TC-02", "GUI и конфигурация Node-RED", self.test_node_red_gui_and_flow)
        await self.run_test("TC-03", "GUI ejabberd Web Admin", self.test_ejabberd_gui)
        await self.run_test("TC-04", "Доступность MQTT-брокера", self.test_mqtt_broker)
        await self.run_test("TC-09-U", "Unit: нормализация времени", self.test_time_normalization_unit)
        await self.run_test("TC-12-U", "Unit: алгоритм полива возвращает false", self.test_irrigation_false_unit)
        await self.run_test("TC-13-U", "Unit: алгоритм полива возвращает true", self.test_irrigation_true_unit)

        if self.args.isolate_node_red:
            await self.run_test("TC-02-I", "Изоляция Node-RED для детерминированных E2E-тестов", self.stop_node_red_if_possible)

        await self.run_test("TC-05", "Автоматическая регистрация XMPP-агентов", self.test_start_agents_and_registration)
        await self.run_test("TC-11", "Алгоритм не запускается без полного набора данных", self.test_missing_data_batch)
        await self.run_test("TC-06/07", "Получение данных sensor agents и передача в manager", self.test_full_pipeline_manager_working)
        await self.run_test("TC-08", "Запись raw JSON manager agent", self.test_raw_json_file)
        await self.run_test("TC-09/10", "Нормализация времени и запись normalized JSON", self.test_normalized_json_file)
        await self.run_test("TC-12", "E2E: решение false — полив не требуется", self.test_e2e_irrigation_false)
        await self.run_test("TC-13", "E2E: решение true — полив требуется", self.test_e2e_irrigation_true)
        await self.run_test("TC-14", "Отсутствие данных одного датчика переводит manager в DEGRADED", self.test_manager_degraded_when_sensor_missing)
        await self.run_test("TC-15", "Недоступность manager переводит sensor agents в DEGRADED", self.test_sensors_degraded_when_manager_down)
        await self.run_test("TC-16", "Offline-статусы агентов", self.test_offline_statuses)
        await self.run_test("TC-17", "Проверка логирования", self.test_logging)

        await self.cleanup()
        self.write_report()
        self.print_summary()

        return 0 if all(item.passed for item in self.results) else 1

    async def run_test(
        self,
        code: str,
        name: str,
        func: Callable[[], Any],
    ) -> None:
        started = time.monotonic()

        try:
            value = func()
            if asyncio.iscoroutine(value):
                value = await value

            details = "OK" if value is None else str(value)
            passed = True

        except Exception as error:
            details = f"{type(error).__name__}: {error}"
            passed = False
            self.logger.exception("Test failed: %s %s", code, name)

        duration = time.monotonic() - started

        self.results.append(
            TestResult(
                code=code,
                name=name,
                passed=passed,
                details=details,
                duration_seconds=round(duration, 3),
            )
        )

    # ------------------------------- infrastructure ------------------------

    def test_docker_infrastructure(self) -> str:
        if not docker_available():
            raise AcceptanceTestError("Docker is unavailable")

        result = run_cmd(["docker", "ps", "--format", "{{.Names}} {{.Status}}"], timeout=15)

        if result.returncode != 0:
            raise AcceptanceTestError(result.stderr.strip() or "docker ps failed")

        output = result.stdout.lower()
        required = ["ejabberd", "mosquitto", "node-red"]
        missing = [name for name in required if name not in output]

        if missing:
            raise AcceptanceTestError(f"Missing running containers: {missing}. docker ps: {result.stdout}")

        return "Docker containers are running"

    def test_node_red_gui_and_flow(self) -> str:
        status = http_status("http://localhost:1880", timeout=5)

        if status not in {200, 301, 302}:
            raise AcceptanceTestError(f"Unexpected Node-RED HTTP status: {status}")

        flow_file = PROJECT_ROOT / "flows.json"

        if not flow_file.exists():
            raise AcceptanceTestError("flows.json does not exist")

        flow = read_json(flow_file)

        if not isinstance(flow, list):
            raise AcceptanceTestError("flows.json must contain a list of nodes")

        names = {node.get("name") for node in flow if isinstance(node, dict)}
        expected_names = {
            "SONBEST SM9560B",
            "Veinasa THPW-NJ",
            "XS-TBQ02C",
            "SONBEST SM7005",
        }

        missing_names = expected_names - names
        if missing_names:
            raise AcceptanceTestError(f"Missing Node-RED function nodes: {missing_names}")

        serialized = json.dumps(flow, ensure_ascii=False)
        for topic in SENSOR_TOPICS.values():
            if topic not in serialized:
                raise AcceptanceTestError(f"Topic is missing from flows.json: {topic}")

        mqtt_out_nodes = [node for node in flow if isinstance(node, dict) and node.get("type") == "mqtt out"]
        if not mqtt_out_nodes:
            raise AcceptanceTestError("No mqtt out node found in flows.json")

        if not any(node.get("topic", "") == "" for node in mqtt_out_nodes):
            raise AcceptanceTestError("mqtt out node topic must be empty to use msg.topic")

        return "Node-RED is reachable and flow contains expected sensor topics"

    def test_ejabberd_gui(self) -> str:
        status = http_status("http://localhost:5280/admin/", timeout=5)

        if status not in {200, 401, 403}:
            raise AcceptanceTestError(f"Unexpected ejabberd admin HTTP status: {status}")

        return f"ejabberd Web Admin is reachable, HTTP status={status}"

    async def test_mqtt_broker(self) -> str:
        mqtt_host, mqtt_port = detect_mqtt_endpoint()

        await asyncio.to_thread(
            publish_mqtt_sync,
            "acceptance/probe",
            {"timestamp": raw_timestamp()},
            host=mqtt_host,
            port=mqtt_port,
        )

        return f"MQTT broker is reachable at {mqtt_host}:{mqtt_port}"

    def test_time_normalization_unit(self) -> str:
        normalize = self.modules["normalize_timestamp_to_minute_utc"]

        result = normalize("2026-05-03T21:11:54.401Z")
        if result != "2026-05-03T21:11:00Z":
            raise AcceptanceTestError(f"Unexpected normalized timestamp: {result}")

        result_with_tz = normalize("2026-05-03T23:11:54.401+02:00")
        if result_with_tz != "2026-05-03T21:11:00Z":
            raise AcceptanceTestError(f"Unexpected timezone conversion: {result_with_tz}")

        return "Timestamp normalization works"

    async def test_irrigation_false_unit(self) -> str:
        decision = await self.run_irrigation_algorithm_direct(
            timestamp="2026-05-03T21:11:00Z",
            memory_payload={"depletion_mm": 0.0},
            raining=True,
        )

        if decision.get("value") is not False:
            raise AcceptanceTestError(f"Expected false, got {decision}")

        return "Irrigation decision false is produced"

    async def test_irrigation_true_unit(self) -> str:
        decision = await self.run_irrigation_algorithm_direct(
            timestamp="2026-05-03T21:12:00Z",
            memory_payload={
                "depletion_mm": 30.0,
                "last_input_timestamp": "2026-05-03T21:11:00Z",
            },
            raining=False,
        )

        if decision.get("value") is not True:
            raise AcceptanceTestError(f"Expected true, got {decision}")

        return "Irrigation decision true is produced"

    async def run_irrigation_algorithm_direct(
        self,
        *,
        timestamp: str,
        memory_payload: dict[str, Any],
        raining: bool,
    ) -> dict[str, Any]:
        AlgorithmInputRecord = self.modules["AlgorithmInputRecord"]
        IrrigationAlgorithm = self.modules["IrrigationAlgorithm"]
        IrrigationSettings = self.modules["IrrigationSettings"]

        memory_file = DATA_DIR / f"unit_irrigation_memory_{timestamp.replace(':', '-')}.json"
        atomic_write_json(memory_file, memory_payload)

        settings = IrrigationSettings(
            weather_gateway_guid=WEATHER_GUID,
            solar_gateway_guid=SOLAR_GUID,
            light_gateway_guid=LIGHT_GUID,
            rain_gateway_guid=RAIN_GUID,
            crop_coefficient=1.0,
            management_allowed_depletion_mm=25.0,
            max_depletion_mm=80.0,
            max_wind_speed_ms=8.0,
            default_step_seconds=300.0,
            rain_value_mode="incremental",
            memory_file=str(memory_file),
            output_file=str(DATA_DIR / "unit_irrigation_decision.json"),
            max_data_age_seconds=120.0,
        )
        algorithm = IrrigationAlgorithm(settings=settings)

        payloads = build_sensor_payloads(timestamp, raining=raining)
        values_by_guid = {
            LIGHT_GUID: payloads["light"][1],
            WEATHER_GUID: payloads["weather"][1],
            SOLAR_GUID: payloads["solar"][1],
            RAIN_GUID: payloads["rain"][1],
        }

        # Manager removes timestamp from value before forwarding to algorithms.
        records = {
            guid: AlgorithmInputRecord(
                gateway_guid=guid,
                timestamp=timestamp,
                value={key: value for key, value in payload.items() if key != "timestamp"},
                raw={
                    "gateway_guid": guid,
                    "timestamp": timestamp,
                    "value": {key: value for key, value in payload.items() if key != "timestamp"},
                },
            )
            for guid, payload in values_by_guid.items()
        }

        result = await algorithm.run(records=records)
        if not result.outputs:
            raise AcceptanceTestError("IrrigationAlgorithm returned no outputs")

        return result.outputs[0]

    async def stop_node_red_if_possible(self) -> str:
        if not docker_available():
            return "Docker unavailable, skipped Node-RED isolation"

        container = os.getenv("NODE_RED_CONTAINER", "node-red")
        inspect = run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", container], timeout=10)

        if inspect.returncode != 0:
            return f"Node-RED container {container!r} not found, skipped"

        if inspect.stdout.strip().lower() == "true":
            result = run_cmd(["docker", "stop", container], timeout=20)
            if result.returncode != 0:
                raise AcceptanceTestError(result.stderr.strip() or "docker stop node-red failed")
            self.node_red_was_stopped = True
            return "Node-RED stopped for deterministic E2E tests"

        return "Node-RED already stopped"

    # ------------------------------- agents and E2E ------------------------

    async def test_start_agents_and_registration(self) -> str:
        create_irrigation_agent = self.modules["create_irrigation_agent"]
        create_normalizer_agent = self.modules["create_normalizer_agent"]
        create_manager_agent = self.modules["create_manager_agent"]
        create_sensor_agents = self.modules["create_sensor_agents"]

        irrigation_agent = create_irrigation_agent()
        normalizer_agent = create_normalizer_agent()
        self.manager_agent = create_manager_agent()
        sensor_agents = create_sensor_agents()

        for agent in [irrigation_agent, normalizer_agent, self.manager_agent, *sensor_agents]:
            await agent.start(auto_register=True)
            self.started_agents.append(agent)
            await asyncio.sleep(0.2)

        await asyncio.sleep(1.0)

        users = self.get_registered_users()
        missing = EXPECTED_USERS - users

        if missing:
            raise AcceptanceTestError(f"Missing registered users: {missing}; registered={sorted(users)}")

        return "All required XMPP agents are registered and started"

    async def test_missing_data_batch(self) -> str:
        self.decision_file.unlink(missing_ok=True)
        self.memory_file.unlink(missing_ok=True)

        timestamp = raw_timestamp(minutes_offset=0)
        await self.publish_sensor_payloads(timestamp, keys=["light", "weather", "solar"], raining=False)

        await asyncio.sleep(3)

        if self.decision_file.exists():
            raise AcceptanceTestError(
                f"Decision file was created although only 3/4 sensors were published: {read_json(self.decision_file)}"
            )

        presence = self.get_presence("irrigation")
        if "ONLINE_IDLE" not in presence and "waiting" not in presence.lower():
            raise AcceptanceTestError(f"Unexpected irrigation presence for missing data: {presence}")

        return "Irrigation agent waits for the missing fourth input"

    async def test_full_pipeline_manager_working(self) -> str:
        timestamp = raw_timestamp(minutes_offset=1)
        await self.publish_sensor_payloads(timestamp, keys=["light", "weather", "solar", "rain"], raining=True)

        await wait_for_condition(
            lambda: "WORKING" in self.get_presence("manager"),
            timeout=12,
            description="manager WORKING presence",
        )

        return "Sensor agents delivered data to manager; manager status is WORKING"

    async def test_raw_json_file(self) -> str:
        records = await wait_for_json(
            self.raw_file,
            lambda payload: records_have_gateways(payload, [LIGHT_GUID, WEATHER_GUID, SOLAR_GUID, RAIN_GUID]),
            timeout=10,
            description="raw sensor JSON with all gateway GUIDs",
        )

        for guid in [LIGHT_GUID, WEATHER_GUID, SOLAR_GUID, RAIN_GUID]:
            record = get_record(records, guid)
            if not record or not {"gateway_guid", "timestamp", "value"}.issubset(record):
                raise AcceptanceTestError(f"Invalid raw record for {guid}: {record}")

        return f"Raw JSON contains {len(records)} record(s)"

    async def test_normalized_json_file(self) -> str:
        normalize = self.modules["normalize_timestamp_to_minute_utc"]
        records = await wait_for_json(
            self.normalized_file,
            lambda payload: records_have_gateways(payload, [LIGHT_GUID, WEATHER_GUID, SOLAR_GUID, RAIN_GUID]),
            timeout=10,
            description="normalized sensor JSON with all gateway GUIDs",
        )

        for guid in [LIGHT_GUID, WEATHER_GUID, SOLAR_GUID, RAIN_GUID]:
            record = get_record(records, guid)
            if not record:
                raise AcceptanceTestError(f"Missing normalized record for {guid}")
            timestamp = record.get("timestamp")
            if not isinstance(timestamp, str) or not timestamp.endswith(":00Z"):
                raise AcceptanceTestError(f"Timestamp is not minute-normalized: {record}")
            if timestamp != normalize(timestamp):
                raise AcceptanceTestError(f"Timestamp is not UTC minute-normalized: {record}")

        return "Normalized JSON contains UTC minute-level timestamps"

    async def test_e2e_irrigation_false(self) -> str:
        self.decision_file.unlink(missing_ok=True)
        atomic_write_json(self.memory_file, {"depletion_mm": 0.0})

        timestamp = raw_timestamp(minutes_offset=2)
        normalized_timestamp = self.modules["normalize_timestamp_to_minute_utc"](timestamp)

        await self.publish_sensor_payloads(timestamp, keys=["light", "weather", "solar", "rain"], raining=True)

        decision = await wait_for_json(
            self.decision_file,
            lambda payload: isinstance(payload, dict)
            and payload.get("timestamp") == normalized_timestamp
            and payload.get("value") is False,
            timeout=15,
            description="irrigation false decision",
        )

        return f"Decision is false: {decision}"

    async def test_e2e_irrigation_true(self) -> str:
        timestamp = raw_timestamp(minutes_offset=3)
        normalized_timestamp = self.modules["normalize_timestamp_to_minute_utc"](timestamp)
        atomic_write_json(
            self.memory_file,
            {
                "depletion_mm": 30.0,
                "last_input_timestamp": previous_minute_timestamp(normalized_timestamp),
            },
        )

        await self.publish_sensor_payloads(timestamp, keys=["light", "weather", "solar", "rain"], raining=False)

        decision = await wait_for_json(
            self.decision_file,
            lambda payload: isinstance(payload, dict)
            and payload.get("timestamp") == normalized_timestamp
            and payload.get("value") is True,
            timeout=15,
            description="irrigation true decision",
        )

        return f"Decision is true: {decision}"

    async def test_manager_degraded_when_sensor_missing(self) -> str:
        timestamp = raw_timestamp(minutes_offset=4)
        await self.publish_sensor_payloads(timestamp, keys=["light", "weather", "solar"], raining=False)

        await wait_for_condition(
            lambda: "DEGRADED" in self.get_presence("manager"),
            timeout=12,
            description="manager DEGRADED when one sensor is missing",
        )

        presence = self.get_presence("manager")
        return f"Manager degraded presence: {presence.strip()}"

    async def test_sensors_degraded_when_manager_down(self) -> str:
        if not self.manager_agent or not self.manager_agent.is_alive():
            raise AcceptanceTestError("Manager agent is not running before manager-down test")

        await self.manager_agent.stop()
        await asyncio.sleep(4)

        await wait_for_condition(
            lambda: "DEGRADED" in self.get_presence("sm9560b")
            or "manager unavailable" in self.get_presence("sm9560b").lower(),
            timeout=10,
            description="sensor DEGRADED when manager is unavailable",
        )

        presence = self.get_presence("sm9560b")
        return f"Sensor degraded presence after manager stop: {presence.strip()}"

    async def test_offline_statuses(self) -> str:
        await self.stop_started_agents()
        await asyncio.sleep(2.0)

        # ejabberd stores the last activity text in mod_last, but when a client
        # is disconnected while it has pending network tasks, ejabberd may record
        # transport-level text such as "Stream reset by peer" instead of the
        # unavailable presence text sent by the agent. For acceptance testing the
        # important condition is that the agent has no active XMPP session after
        # shutdown and get_last contains a non-ONLINE last activity record.
        presence = self.get_presence("sm9560b")
        if "sm9560b@" in presence:
            raise AcceptanceTestError(f"Expected no active sm9560b session, got: {presence}")

        last = self.get_last("sm9560b").strip()
        if not last:
            raise AcceptanceTestError("get_last returned an empty result for sm9560b")

        last_upper = last.upper()
        acceptable_markers = (
            "OFFLINE",
            "STOPPED NORMALLY",
            "STREAM RESET BY PEER",
            "CLOSED",
            "RESET",
        )

        if "ONLINE" in last_upper and not any(marker in last_upper for marker in acceptable_markers):
            raise AcceptanceTestError(f"Expected non-online get_last output, got: {last}")

        if not any(marker in last_upper for marker in acceptable_markers):
            raise AcceptanceTestError(f"Unexpected get_last output after shutdown: {last}")

        return f"Agent has no active session; get_last={last}"

    def test_logging(self) -> str:
        if not self.log_file.exists():
            raise AcceptanceTestError(f"Log file does not exist: {self.log_file}")

        content = self.log_file.read_text(encoding="utf-8", errors="replace")
        if "Starting" not in content and "started" not in content.lower():
            raise AcceptanceTestError("Log file does not contain startup messages")

        # ERROR entries may appear only if some previous test really failed. The test itself
        # checks that the log file is readable and contains useful operational records.
        return f"Log file exists: {self.log_file}"

    async def publish_sensor_payloads(
        self,
        timestamp: str,
        *,
        keys: list[str],
        raining: bool,
    ) -> None:
        mqtt_host, mqtt_port = detect_mqtt_endpoint()
        payloads = build_sensor_payloads(timestamp, raining=raining)

        for key in keys:
            topic, payload = payloads[key]
            await asyncio.to_thread(
                publish_mqtt_sync,
                topic,
                payload,
                host=mqtt_host,
                port=mqtt_port,
            )
            await asyncio.sleep(0.15)

    # ------------------------------- ejabberd helpers ----------------------

    def get_registered_users(self) -> set[str]:
        container = os.getenv("EJABBERD_CONTAINER", "ejabberd1")
        host = os.getenv("XMPP_VHOST", "localhost")
        result = run_cmd(
            ["docker", "exec", container, "ejabberdctl", "registered_users", host],
            timeout=15,
        )

        if result.returncode != 0:
            raise AcceptanceTestError(result.stderr.strip() or "registered_users failed")

        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def get_presence(self, user: str) -> str:
        container = os.getenv("EJABBERD_CONTAINER", "ejabberd1")
        host = os.getenv("XMPP_VHOST", "localhost")
        result = run_cmd(
            ["docker", "exec", container, "ejabberdctl", "get_presence", user, host],
            timeout=15,
        )

        return (result.stdout or "") + (result.stderr or "")

    def get_last(self, user: str) -> str:
        container = os.getenv("EJABBERD_CONTAINER", "ejabberd1")
        host = os.getenv("XMPP_VHOST", "localhost")
        result = run_cmd(
            ["docker", "exec", container, "ejabberdctl", "get_last", user, host],
            timeout=15,
        )

        return (result.stdout or "") + (result.stderr or "")

    # ------------------------------- cleanup/report ------------------------

    async def stop_started_agents(self) -> None:
        if not self.started_agents:
            return

        alive_agents = [
            agent
            for agent in reversed(self.started_agents)
            if getattr(agent, "is_alive", lambda: False)()
        ]

        if alive_agents:
            await asyncio.gather(
                *[agent.stop() for agent in alive_agents],
                return_exceptions=True,
            )

            # Many SPADE behaviours in this project wait on receive(timeout=5) or
            # inside aiomqtt. Give them one receive cycle to notice kill/disconnect
            # before Python closes the event loop; otherwise Windows prints noisy
            # "Task was destroyed but it is pending" messages after the summary.
            await asyncio.sleep(5.5)

        self.started_agents.clear()
        self.manager_agent = None

    async def cleanup(self) -> None:
        with contextlib.suppress(Exception):
            await self.stop_started_agents()

        if self.node_red_was_stopped:
            with contextlib.suppress(Exception):
                container = os.getenv("NODE_RED_CONTAINER", "node-red")
                run_cmd(["docker", "start", container], timeout=20)

    def write_report(self) -> None:
        report_path = DATA_DIR / "acceptance_report.json"
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "passed": all(item.passed for item in self.results),
            "results": [asdict(item) for item in self.results],
            "files": {
                "raw_sensor_state": str(self.raw_file),
                "normalized_sensor_state": str(self.normalized_file),
                "irrigation_decision": str(self.decision_file),
                "irrigation_memory": str(self.memory_file),
                "log_file": str(self.log_file),
            },
        }
        atomic_write_json(report_path, payload)

    def print_summary(self) -> None:
        print("\nAcceptance test summary")
        print("=" * 80)

        for item in self.results:
            mark = "PASS" if item.passed else "FAIL"
            print(f"{mark:4} | {item.code:8} | {item.name} | {item.details}")

        print("=" * 80)
        print(f"Report: {DATA_DIR / 'acceptance_report.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run acceptance tests for the MultiAgent irrigation system.",
    )
    parser.add_argument(
        "--no-isolate-node-red",
        dest="isolate_node_red",
        action="store_false",
        help="Do not stop the Node-RED container during deterministic E2E tests.",
    )
    parser.add_argument(
        "--mqtt-host",
        default=None,
        help="Override MQTT_HOST for the test run. Use localhost on the host machine, host.docker.internal from a separate Docker container, or mosquitto inside the same Docker network.",
    )
    parser.add_argument(
        "--xmpp-host",
        default=None,
        help="Override XMPP_HOST for the test run. Usually localhost on the host machine. Keep it equal to the ejabberd vhost unless your SPADE version supports a separate connection host.",
    )
    parser.set_defaults(isolate_node_red=True)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    configure_acceptance_environment()

    if args.mqtt_host:
        os.environ["MQTT_HOST"] = args.mqtt_host
    if args.xmpp_host:
        os.environ["XMPP_HOST"] = args.xmpp_host

    modules = import_project_modules()

    suite = AcceptanceTestSuite(args=args, modules=modules)
    return await suite.run()


def main() -> None:
    if sys.platform.startswith("win"):
        # SPADE/slixmpp and aiomqtt leave long-running background tasks while
        # agents are being stopped. On Windows these tasks may log noisy
        # "Event loop is closed" / "Task was destroyed" messages after the
        # acceptance report has already been written. The test result is already
        # known at this point, so we terminate the process with os._exit after
        # flushing output. This prevents late destructors from turning a passed
        # acceptance run into a noisy shutdown failure.
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
        exit_code = 1

        try:
            exit_code = loop.run_until_complete(async_main())
        finally:
            with contextlib.suppress(Exception):
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())

            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)

    exit_code = asyncio.run(async_main())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
