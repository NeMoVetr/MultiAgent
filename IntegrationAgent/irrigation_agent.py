import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.template import Template

from StatusAgent import AgentStatus, StatusAwareAgent
from algoritms import (
    IrrigationAlgorithm,
    IrrigationAlgorithmConfig,
    IrrigationDecision,
    SensorRecord,
)

load_dotenv()

logger = logging.getLogger(__name__)

ALGORITHM_INPUT_ONTOLOGY = "algorithm-input"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s. Using default: %s", name, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def make_algorithm_input_template() -> Template:
    template = Template()
    template.set_metadata("ontology", ALGORITHM_INPUT_ONTOLOGY)
    template.set_metadata("performative", "inform")
    return template


class AlgorithmInputReceiverBehaviour(CyclicBehaviour):
    async def on_start(self) -> None:
        logger.info("Irrigation algorithm input receiver started")

    async def run(self) -> None:
        msg = await self.receive(timeout=5)

        if msg is None:
            return

        try:
            body = json.loads(msg.body)
        except json.JSONDecodeError:
            logger.warning("Irrigation agent received invalid JSON")
            return

        record = self.parse_record(body)

        if record is None:
            logger.warning("Irrigation agent received invalid input record")
            return

        self.agent.latest_records[record.gateway_guid] = record
        self.agent.updated_gateways.add(record.gateway_guid)

        logger.debug(
            "Irrigation agent received data from gateway %s",
            record.gateway_guid,
        )

        await self.try_run_algorithm()

    def parse_record(self, body: dict[str, Any]) -> SensorRecord | None:
        gateway_guid = body.get("gateway_guid")
        timestamp = body.get("timestamp")
        value = body.get("value")

        if not gateway_guid or not timestamp:
            return None

        return SensorRecord(
            gateway_guid=str(gateway_guid),
            timestamp=str(timestamp),
            value=value,
            received_monotonic=time.monotonic(),
        )

    async def try_run_algorithm(self) -> None:
        missing = self.agent.algorithm.missing_gateways(
            self.agent.latest_records,
        )

        if missing:
            self.agent.set_agent_status(
                AgentStatus.ONLINE_IDLE,
                f"waiting for {len(missing)} input(s)",
                priority=8,
            )

            logger.debug(
                "Irrigation algorithm is waiting for missing gateways: %s",
                missing,
            )

            return

        require_all_updated = env_bool("ALGORITHM_REQUIRE_ALL_UPDATED", True)

        if require_all_updated:
            not_updated = [
                gateway_guid
                for gateway_guid in self.agent.required_gateway_guids
                if gateway_guid not in self.agent.updated_gateways
            ]

            if not_updated:
                self.agent.set_agent_status(
                    AgentStatus.ONLINE_IDLE,
                    "waiting for complete input batch",
                    priority=8,
                )

                logger.debug(
                    "Irrigation algorithm is waiting for updated gateways: %s",
                    not_updated,
                )

                return

        max_age = env_float("ALGORITHM_MAX_DATA_AGE_SECONDS", 120.0)
        stale = self.agent.algorithm.stale_gateways(
            self.agent.latest_records,
            max_age_seconds=max_age,
        )

        if stale:
            self.agent.set_agent_status(
                AgentStatus.DEGRADED,
                "stale input data",
                priority=8,
            )

            logger.warning(
                "Irrigation algorithm cannot run because data is stale: %s",
                stale,
            )

            return

        try:
            decision = self.agent.algorithm.run_from_records(
                self.agent.latest_records,
            )

            self.agent.write_decision(decision)

            if decision.value:
                self.agent.set_agent_status(
                    AgentStatus.WORKING,
                    "irrigation required",
                    priority=8,
                )
            else:
                self.agent.set_agent_status(
                    AgentStatus.WORKING,
                    "irrigation not required",
                    priority=8,
                )

            self.agent.set_offline_message("stopped normally")
            self.agent.updated_gateways.clear()
            self.agent.last_successful_run_monotonic = time.monotonic()

            logger.info(
                "Irrigation algorithm completed. Decision=%s",
                decision.value,
            )

        except Exception:
            self.agent.set_agent_status(
                AgentStatus.DEGRADED,
                "algorithm error",
                priority=8,
            )

            logger.exception("Irrigation algorithm failed")


class AlgorithmHealthMonitorBehaviour(PeriodicBehaviour):
    async def run(self) -> None:
        if self.agent.last_successful_run_monotonic is None:
            return

        max_silence = env_float("ALGORITHM_MANAGER_SILENCE_SECONDS", 180.0)
        age = time.monotonic() - self.agent.last_successful_run_monotonic

        if age > max_silence:
            self.agent.set_agent_status(
                AgentStatus.DEGRADED,
                "no recent complete input data",
                priority=8,
            )


class IrrigationAgent(StatusAwareAgent):
    async def setup(self) -> None:
        logger.info("Irrigation agent started: %s", self.jid)

        config = IrrigationAlgorithmConfig.from_env()
        self.algorithm = IrrigationAlgorithm(config=config)

        self.required_gateway_guids = self.algorithm.required_gateway_guids
        self.latest_records: dict[str, SensorRecord] = {}
        self.updated_gateways: set[str] = set()
        self.last_successful_run_monotonic: float | None = None

        self.decision_file = Path(
            os.getenv("IRRIGATION_DECISION_FILE", "./data/irrigation_decision.json")
        )
        self.decision_file.parent.mkdir(parents=True, exist_ok=True)

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            "waiting for sensor data",
            priority=8,
        )
        self.set_offline_message("stopped normally")

        self.add_behaviour(
            AlgorithmInputReceiverBehaviour(),
            make_algorithm_input_template(),
        )

        health_period = env_float("ALGORITHM_HEALTH_CHECK_SECONDS", 10.0)
        self.add_behaviour(AlgorithmHealthMonitorBehaviour(period=health_period))

        logger.info(
            "Irrigation agent configured. Required gateways: %s",
            ", ".join(self.required_gateway_guids),
        )

    def write_decision(self, decision: IrrigationDecision) -> None:
        output = {
            "timestamp": decision.timestamp,
            "value": decision.value,
        }

        tmp_file = self.decision_file.with_name(f".{self.decision_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.decision_file)