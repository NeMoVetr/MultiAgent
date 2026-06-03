import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from DataQualityAgent.data_quality_agent import (
    QUALITY_DATA_ONTOLOGY,
    DataQualityAgent,
    sensor_input_template,
)
from IrrigatorAgent.irrigator_agent import IrrigatorAgent, quality_input_template
from SensorAgent import (
    OpticalRainGaugeAgent,
    SM9560BAgent,
    TBQ02CAgent,
    THPWNJAgent,
    TR4H01XAgent,
    XM8504Agent,
)
from SensorAgent.sensor_agent import SENSOR_DATA_ONTOLOGY
from StatusAgent import AgentStatus
from algorithms.data_quality import DataQuality
from algorithms.irrigator import Irrigation


EXPECTED_SENSOR_NAMES = [
    "Veinasa_THPW_NJ",
    "SONBEST_SM9560B",
    "XS_TBQ02C",
    "SONBEST_XM8504",
    "TR_4H01X",
]


class FakeSender:
    def __init__(self) -> None:
        self.sent_messages = []

    async def send(self, msg) -> None:
        self.sent_messages.append(msg)


class FailingSender:
    async def send(self, msg) -> None:
        raise RuntimeError("simulated send failure")


class FakeBehaviour:
    def __init__(self) -> None:
        self.sent_messages = []

    async def send(self, msg) -> None:
        self.sent_messages.append(msg)

    def is_killed(self) -> bool:
        return False


class FakeMQTTMessage:
    def __init__(self, topic: str, payload: dict[str, Any]) -> None:
        self.topic = topic
        self.payload = json.dumps(payload).encode("utf-8")


class FakeMQTTClient:
    published_messages = []
    client_kwargs = []

    def __init__(self, **kwargs) -> None:
        self.client_kwargs.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def publish(self, topic, payload, qos=0) -> None:
        self.published_messages.append(
            {
                "topic": topic,
                "payload": payload,
                "qos": qos,
            }
        )


def run(coro):
    return asyncio.run(coro)


def make_data_quality_agent(tmp_path: Path) -> DataQualityAgent:
    agent = DataQualityAgent(
        "data_quality@localhost",
        "secret",
        verify_security=False,
    )
    agent.state_lock = asyncio.Lock()
    agent.irrigator_agent_jid = "irrigator@localhost"
    agent.expected_sensor_names = list(EXPECTED_SENSOR_NAMES)
    agent.latest_raw_by_sensor = {}
    agent.latest_clean_by_sensor = {}
    agent.latest_clean_values = {}
    agent.processed_count = 0
    agent.last_processed_monotonic = None
    agent.last_error = None
    agent.data_quality = DataQuality(
        state_path=tmp_path / "data_quality_state.json",
        autosave=True,
    )
    agent.output_file = tmp_path / "sensors_state.json"
    return agent


def make_irrigator_agent(tmp_path: Path) -> IrrigatorAgent:
    agent = IrrigatorAgent(
        "irrigator@localhost",
        "secret",
        verify_security=False,
    )
    agent.state_lock = asyncio.Lock()
    agent.expected_sensor_names = list(EXPECTED_SENSOR_NAMES)
    agent.latest_by_sensor = {}
    agent.latest_values = {}
    agent.last_decision_at_monotonic = None
    agent.last_error = None
    agent.last_decision = None
    agent.output_file = tmp_path / "irrigation_decision.json"
    agent.latitude = 48.708
    agent.longitude = 44.514
    agent.elevation = 100.0
    return agent


def sample(
    sensor_name: str,
    values: dict[str, Any],
    *,
    timestamp: str = "2026-07-15T10:00:00Z",
    is_anomaly: bool = False,
    anomaly_type: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "sensor_sample",
        "sensor_id": sensor_name.lower(),
        "sensor_name": sensor_name,
        "source_name": sensor_name,
        "timestamp": timestamp,
        "values": values,
        "context": {
            "communication_ok": True,
            "sun_state": "day",
            "polling_interval_minutes": 10,
            "probes_same_depth": True,
            "rain_or_irrigation": False,
        },
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    }


def complete_quality_records(timestamp: str = "2026-07-15T10:00:00Z") -> list[dict[str, Any]]:
    return [
        {
            "sensor_name": "Veinasa_THPW_NJ",
            "timestamp": timestamp,
            "values": {
                "air_temperature_c": 30.1,
                "air_humidity_percent": 33.0,
                "air_pressure_hpa": 1005.0,
                "wind_speed_ms": 1.8,
                "wind_direction_deg": 180.0,
            },
        },
        {
            "sensor_name": "SONBEST_SM9560B",
            "timestamp": timestamp,
            "values": {"illuminance_lux": 42000.0},
        },
        {
            "sensor_name": "XS_TBQ02C",
            "timestamp": timestamp,
            "values": {"solar_radiation_wm2": 760.0},
        },
        {
            "sensor_name": "SONBEST_XM8504",
            "timestamp": timestamp,
            "values": {"rain_interval_mm": 0.0},
        },
        {
            "sensor_name": "TR_4H01X",
            "timestamp": timestamp,
            "values": {
                "soil_moisture_probe_1_percent": 14.0,
                "soil_moisture_probe_2_percent": 13.5,
                "soil_moisture_probe_3_percent": 14.2,
                "soil_moisture_probe_4_percent": 13.9,
            },
        },
    ]


# 1. Module tests for algorithms


def test_algorithm_modules_data_quality_output_feeds_irrigation_decision(tmp_path: Path):
    dq = DataQuality(state_path=tmp_path / "dq_state.json", autosave=True)
    values: dict[str, float] = {}

    for body in [
        sample(
            "Veinasa_THPW_NJ",
            {
                "air_temperature_c": 30.1,
                "air_humidity_percent": 33.0,
                "air_pressure_hpa": 1005.0,
                "wind_speed_ms": 1.8,
                "wind_direction_deg": 180.0,
            },
        ),
        sample("XS_TBQ02C", {"solar_radiation_wm2": 760.0}),
        sample("SONBEST_XM8504", {"rain_interval_mm": 0.0}),
        sample(
            "TR_4H01X",
            {
                "soil_moisture_probe_1_percent": 14.0,
                "soil_moisture_probe_2_percent": 13.5,
                "soil_moisture_probe_3_percent": 14.2,
                "soil_moisture_probe_4_percent": 13.9,
            },
        ),
    ]:
        values.update(
            dq.clean(
                body["sensor_name"],
                body["values"],
                body["context"],
            )
        )

    decision = Irrigation(
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
    ).get_decision()

    assert isinstance(decision, bool)


# 4. SPADE agent integration tests


def test_spade_message_templates_match_pipeline_ontologies():
    sensor_template = sensor_input_template()
    quality_template = quality_input_template()

    assert sensor_template.metadata["ontology"] == SENSOR_DATA_ONTOLOGY
    assert sensor_template.metadata["performative"] == "inform"
    assert quality_template.metadata["ontology"] == QUALITY_DATA_ONTOLOGY
    assert quality_template.metadata["performative"] == "inform"


def test_data_quality_agent_forwards_spade_message_to_irrigator(tmp_path: Path):
    agent = make_data_quality_agent(tmp_path)
    sender = FakeSender()

    run(
        agent.process_sensor_sample(
            sample("SONBEST_SM9560B", {"illuminance_lux": 90000.0}, is_anomaly=True),
            sender,
        )
    )

    assert len(sender.sent_messages) == 1
    msg = sender.sent_messages[0]
    assert str(msg.to).split("/")[0] == "irrigator@localhost"
    assert msg.get_metadata("ontology") == QUALITY_DATA_ONTOLOGY

    body = json.loads(msg.body)
    assert body["sensor_name"] == "SONBEST_SM9560B"
    assert body["source_anomaly"] is True
    assert "illuminance_lux" in body["values"]


# 5. Data quality tests


@pytest.mark.parametrize(
    "sensor_name, values, expected_key, expected_value",
    [
        (
            "Veinasa_THPW_NJ",
            {
                "air_temperature_c": 130,
                "air_humidity_percent": 140,
                "air_pressure_hpa": 1300,
                "wind_speed_ms": 80,
                "wind_direction_deg": 720,
            },
            "air_temperature_c",
            0.0,
        ),
        ("SONBEST_SM9560B", {"illuminance_lux": 90000}, "illuminance_lux", 0.0),
        ("XS_TBQ02C", {"solar_radiation_wm2": -15}, "solar_radiation_wm2", 0.0),
        ("SONBEST_XM8504", {"rain_interval_mm": -1}, "rain_interval_mm", 0.0),
        (
            "TR_4H01X",
            {
                "soil_moisture_probe_1_percent": 140,
                "soil_moisture_probe_2_percent": 24,
                "soil_moisture_probe_3_percent": 24,
                "soil_moisture_probe_4_percent": 24,
            },
            "soil_moisture_probe_1_percent",
            0.0,
        ),
    ],
)
def test_data_quality_replaces_sensor_anomalies(
    tmp_path: Path,
    sensor_name: str,
    values: dict[str, Any],
    expected_key: str,
    expected_value: float,
):
    dq = DataQuality(state_path=tmp_path / "dq_state.json", autosave=True)
    cleaned = dq.clean(
        sensor_name,
        values,
        {
            "communication_ok": True,
            "sun_state": "day",
            "polling_interval_minutes": 10,
            "probes_same_depth": True,
            "rain_or_irrigation": False,
        },
    )

    assert cleaned[expected_key] == pytest.approx(expected_value)


# 6. Agent status tests


def test_sensor_status_changes_to_working_and_degraded_from_mqtt_payloads():
    agent = SM9560BAgent("sm9560b@localhost", "secret", verify_security=False)
    agent.set("data_quality_agent_jid", "data_quality@localhost")
    agent.set("mqtt_topic", "rs485/sm9560b")
    behaviour = FakeBehaviour()

    run(
        agent.handle_mqtt_message(
            behaviour,
            FakeMQTTMessage(
                "rs485/sm9560b",
                {
                    "sensorId": "sm9560b-1",
                    "timestamp": "2026-07-15T10:00:00Z",
                    "registers": {"illuminance_lux": 1000.0},
                },
            ),
        )
    )
    assert agent.get("last_agent_status") == AgentStatus.WORKING.value
    assert len(behaviour.sent_messages) == 1

    run(
        agent.handle_mqtt_message(
            behaviour,
            FakeMQTTMessage(
                "rs485/sm9560b",
                {
                    "sensorId": "sm9560b-1",
                    "timestamp": "2026-07-15T10:01:00Z",
                    "registers": {"unexpected": 123.0},
                },
            ),
        )
    )
    assert agent.get("last_agent_status") == AgentStatus.DEGRADED.value
    assert len(behaviour.sent_messages) == 1


def test_data_quality_and_irrigator_statuses_wait_then_work(tmp_path: Path):
    dq_agent = make_data_quality_agent(tmp_path)
    irrigator = make_irrigator_agent(tmp_path)

    run(dq_agent.refresh_presence_status())
    run(irrigator.refresh_presence_status())
    assert dq_agent.get("last_agent_status") == AgentStatus.ONLINE_IDLE.value
    assert irrigator.get("last_agent_status") == AgentStatus.ONLINE_IDLE.value

    for record in complete_quality_records():
        run(irrigator.process_quality_sample(record))

    assert irrigator.get("last_agent_status") == AgentStatus.WORKING.value


# 7. End-to-end tests


def test_end_to_end_pipeline_writes_clean_state_and_decision_json(tmp_path: Path):
    dq_agent = make_data_quality_agent(tmp_path)
    irrigator = make_irrigator_agent(tmp_path)
    sender = FakeSender()

    sensor_samples = [
        sample("SONBEST_SM9560B", {"illuminance_lux": 42000.0}),
        sample(
            "Veinasa_THPW_NJ",
            {
                "air_temperature_c": 30.1,
                "air_humidity_percent": 33.0,
                "air_pressure_hpa": 1005.0,
                "wind_speed_ms": 1.8,
                "wind_direction_deg": 180.0,
            },
        ),
        sample("XS_TBQ02C", {"solar_radiation_wm2": 760.0}),
        sample("SONBEST_XM8504", {"rain_interval_mm": 0.0}),
        sample(
            "TR_4H01X",
            {
                "soil_moisture_probe_1_percent": 14.0,
                "soil_moisture_probe_2_percent": 13.5,
                "soil_moisture_probe_3_percent": 14.2,
                "soil_moisture_probe_4_percent": 13.9,
            },
        ),
    ]

    for sensor_body in sensor_samples:
        run(dq_agent.process_sensor_sample(sensor_body, sender))
        forwarded = json.loads(sender.sent_messages[-1].body)
        run(irrigator.process_quality_sample(forwarded))

    clean_payload = json.loads((tmp_path / "sensors_state.json").read_text(encoding="utf-8"))
    decision_payload = json.loads(
        (tmp_path / "irrigation_decision.json").read_text(encoding="utf-8")
    )

    assert len(clean_payload) == 5
    assert isinstance(decision_payload["value"], bool)
    assert "soil_moisture_probe_1_percent" in decision_payload["inputs"]


def test_irrigator_publishes_decision_to_configured_mqtt_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    FakeMQTTClient.published_messages = []
    FakeMQTTClient.client_kwargs = []
    monkeypatch.setattr(
        "IrrigatorAgent.irrigator_agent.aiomqtt.Client",
        FakeMQTTClient,
    )

    agent = make_irrigator_agent(tmp_path)
    agent.mqtt_host = "mqtt.local"
    agent.mqtt_port = 1884
    agent.mqtt_keepalive = 30
    agent.mqtt_qos = 1
    agent.mqtt_decision_topic = "controls/irrigation"

    for record in complete_quality_records():
        run(agent.process_quality_sample(record))

    assert FakeMQTTClient.client_kwargs == [
        {
            "hostname": "mqtt.local",
            "port": 1884,
            "keepalive": 30,
        }
    ]
    assert len(FakeMQTTClient.published_messages) == 1
    published = FakeMQTTClient.published_messages[0]
    decision_payload = json.loads(
        (tmp_path / "irrigation_decision.json").read_text(encoding="utf-8")
    )

    assert published["topic"] == "controls/irrigation"
    assert published["qos"] == 1
    assert json.loads(published["payload"]) == decision_payload


# 8. Fault-tolerance tests


def test_data_quality_agent_degrades_on_forwarding_failure(tmp_path: Path):
    agent = make_data_quality_agent(tmp_path)

    run(
        agent.process_sensor_sample(
            sample("SONBEST_SM9560B", {"illuminance_lux": 42000.0}),
            FailingSender(),
        )
    )

    assert agent.last_error == "simulated send failure"
    assert agent.get("last_agent_status") == AgentStatus.DEGRADED.value


def test_irrigator_degrades_on_invalid_payload_and_does_not_write_decision(tmp_path: Path):
    agent = make_irrigator_agent(tmp_path)

    run(agent.process_quality_sample({"sensor_name": "TR_4H01X", "values": "bad"}))

    assert agent.last_error == "quality sample missing sensor_name or values"
    assert agent.get("last_agent_status") == AgentStatus.DEGRADED.value
    assert not (tmp_path / "irrigation_decision.json").exists()


def test_irrigator_waits_when_one_sensor_is_missing(tmp_path: Path):
    agent = make_irrigator_agent(tmp_path)

    for record in complete_quality_records()[:-1]:
        run(agent.process_quality_sample(record))

    assert agent.has_complete_input() is False
    assert agent.get("last_agent_status") == AgentStatus.ONLINE_IDLE.value
    assert not (tmp_path / "irrigation_decision.json").exists()


# 9. Regression tests


def test_regression_main_factories_expose_current_sensor_agents():
    from main import create_sensor_agents

    agents = create_sensor_agents()
    classes = {type(agent) for agent in agents}

    assert len(agents) == 6
    assert classes == {
        OpticalRainGaugeAgent,
        SM9560BAgent,
        THPWNJAgent,
        TBQ02CAgent,
        XM8504Agent,
        TR4H01XAgent,
    }


@pytest.mark.parametrize(
    "agent_class, topic, registers, expected_measurements",
    [
        (
            SM9560BAgent,
            "rs485/sm9560b",
            {"illuminance_lux": 65535.0, "ignored": 1},
            {"illuminance_lux": 65535.0},
        ),
        (
            THPWNJAgent,
            "rs485/thpwnj",
            {
                "air_temperature_c": 25.0,
                "air_humidity_percent": 60.0,
                "air_pressure_hpa": 1010.0,
                "wind_speed_ms": 2.5,
                "wind_direction_deg": 180.0,
                "ignored": 1,
            },
            {
                "air_temperature_c": 25.0,
                "air_humidity_percent": 60.0,
                "air_pressure_hpa": 1010.0,
                "wind_speed_ms": 2.5,
                "wind_direction_deg": 180.0,
            },
        ),
        (
            TBQ02CAgent,
            "rs485/tbq02c",
            {"solar_radiation_wm2": 760.0, "ignored": 1},
            {"solar_radiation_wm2": 760.0},
        ),
        (
            XM8504Agent,
            "rs485/xm8504",
            {"rain_interval_mm": 0.0, "ignored": 1},
            {"rain_interval_mm": 0.0},
        ),
        (
            TR4H01XAgent,
            "rs485/tr4h01x",
            {
                "soil_moisture_probe_1_percent": 14.0,
                "soil_moisture_probe_2_percent": 13.5,
                "soil_moisture_probe_3_percent": 14.2,
                "soil_moisture_probe_4_percent": 13.9,
                "ignored": 1,
            },
            {
                "soil_moisture_probe_1_percent": 14.0,
                "soil_moisture_probe_2_percent": 13.5,
                "soil_moisture_probe_3_percent": 14.2,
                "soil_moisture_probe_4_percent": 13.9,
            },
        ),
        (
            OpticalRainGaugeAgent,
            "rs485/optical-rain",
            {
                "rainfall_total_mm": 3.0,
                "rain_interval_mm": 0.5,
                "rain_intensity_mm_min": 0.1,
                "illuminance_lux": 12000.0,
                "ignored": 1,
            },
            {
                "rainfall_total_mm": 3.0,
                "rain_interval_mm": 0.5,
                "rain_intensity_mm_min": 0.1,
                "illuminance_lux": 12000.0,
            },
        ),
    ],
)
def test_regression_sensor_agents_normalize_mqtt_registers(
    agent_class,
    topic: str,
    registers: dict[str, Any],
    expected_measurements: dict[str, Any],
):
    agent = agent_class(f"{agent_class.__name__.lower()}@localhost", "secret", verify_security=False)
    payload = {
        "sensorId": "sensor-1",
        "timestamp": "2026-07-15T10:00:00Z",
        "registers": registers,
    }

    normalized = agent.normalize_mqtt_payload(topic, json.dumps(payload).encode("utf-8"))

    assert normalized["source"] == "mqtt"
    assert normalized["topic"] == topic
    assert normalized["sensor_id"] == "sensor-1"
    assert normalized["sensor_name"] == agent.SENSOR_NAME
    assert normalized["measurements"] == expected_measurements
    assert normalized["values"] == expected_measurements
