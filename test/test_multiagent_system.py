import asyncio
import json
import random
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
    SM9560BAgent,
    TBQ02CAgent,
    THPWNJAgent,
    TR4H01XAgent,
    XM8504Agent,
)
from SensorAgent.sensor_agent import SENSOR_DATA_ONTOLOGY, SensorSimulationResult
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
        lat=48.708,
        lng=44.514,
        elevation=100.0,
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
            20.0,
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
            24.0,
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


def test_sensor_status_changes_to_working_and_degraded():
    agent = SM9560BAgent("sm9560b@localhost", "secret", verify_security=False)
    agent.sensor_id = "sm9560b"
    agent.source_name = "SONBEST SM9560B"
    agent.status_lock = asyncio.Lock()
    agent.generated_count = 0
    agent.anomaly_count = 0
    agent.last_generated_at = None
    agent.last_generated_monotonic = None
    agent.last_send_error = None

    run(
        agent.record_generation(
            SensorSimulationResult({"illuminance_lux": 1000.0}),
            sent_ok=True,
        )
    )
    assert agent.current_operational_status == AgentStatus.WORKING.value

    run(
        agent.record_generation(
            SensorSimulationResult({"illuminance_lux": 1000.0}),
            sent_ok=False,
            error="send failed",
        )
    )
    assert agent.current_operational_status == AgentStatus.DEGRADED.value


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


def test_regression_main_factories_expose_five_current_sensor_agents():
    from main import create_sensor_agents

    agents = create_sensor_agents()
    classes = {type(agent) for agent in agents}

    assert len(agents) == 5
    assert classes == {
        SM9560BAgent,
        THPWNJAgent,
        TBQ02CAgent,
        XM8504Agent,
        TR4H01XAgent,
    }


def test_regression_sensor_normal_samples_stay_inside_declared_ranges():
    agents = [
        SM9560BAgent("sm9560b@localhost", "secret", verify_security=False),
        THPWNJAgent("thpwnj@localhost", "secret", verify_security=False),
        TBQ02CAgent("tbq02c@localhost", "secret", verify_security=False),
        XM8504Agent("xm8504@localhost", "secret", verify_security=False),
        TR4H01XAgent("tr4h01x@localhost", "secret", verify_security=False),
    ]

    for agent in agents:
        agent.random = random.Random(1)
        agent.should_emit_anomaly = lambda: False

    results = [agent.simulate_values().values for agent in agents]

    assert 0 <= results[0]["illuminance_lux"] <= 65535
    assert -40 <= results[1]["air_temperature_c"] <= 80
    assert 0 <= results[1]["air_humidity_percent"] <= 100
    assert 300 <= results[1]["air_pressure_hpa"] <= 1100
    assert 0 <= results[1]["wind_speed_ms"] <= 45
    assert 0 <= results[1]["wind_direction_deg"] <= 359
    assert 0 <= results[2]["solar_radiation_wm2"] <= 2000
    assert 0 <= results[3]["rain_interval_mm"] <= 9999
    for value in results[4].values():
        assert 0 <= value <= 100
