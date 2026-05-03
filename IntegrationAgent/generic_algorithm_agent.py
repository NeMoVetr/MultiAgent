import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from StatusAgent import AgentStatus, StatusAwareAgent
from algorithms import (
    AlgorithmConfig,
    AlgorithmInputRecord,
    AlgorithmMode,
    AlgorithmResult,
    BaseAlgorithm,
    OutputStorageStrategy,
)

load_dotenv()

logger = logging.getLogger(__name__)


def make_input_template(config: AlgorithmConfig) -> Template:
    template = Template()
    template.set_metadata("ontology", config.input_ontology)
    template.set_metadata("performative", "inform")
    return template


def parse_algorithm_input_record(body: dict[str, Any]) -> AlgorithmInputRecord:
    gateway_guid = body.get("gateway_guid")
    timestamp = body.get("timestamp")

    if "value" in body:
        value = body["value"]
    else:
        value = body

    return AlgorithmInputRecord(
        gateway_guid=str(gateway_guid) if gateway_guid is not None else None,
        timestamp=str(timestamp) if timestamp is not None else None,
        value=value,
        raw=body,
    )


class GenericAlgorithmBehaviour(CyclicBehaviour):
    async def on_start(self) -> None:
        logger.info(
            "Generic algorithm behaviour started: %s",
            self.agent.algorithm.config.name,
        )

    async def run(self) -> None:
        msg = await self.receive(timeout=5)

        if msg is None:
            return

        try:
            body = json.loads(msg.body)
        except json.JSONDecodeError:
            self.agent.set_agent_status(
                AgentStatus.DEGRADED,
                "invalid JSON input",
                priority=self.agent.priority,
            )
            logger.warning(
                "Algorithm %s received invalid JSON",
                self.agent.algorithm.config.name,
            )
            return

        record = parse_algorithm_input_record(body)

        async with self.agent.state_lock:
            await self.agent.handle_record(
                record=record,
                sender_behaviour=self,
            )


class GenericAlgorithmAgent(StatusAwareAgent):
    """
    Универсальный SPADE-агент для запуска любого алгоритма.

    Один и тот же класс можно использовать для разных алгоритмов:

    normalizer@localhost  -> TimeNormalizationAlgorithm
    irrigation@localhost  -> IrrigationAlgorithm
    """

    def __init__(
        self,
        jid: str,
        password: str,
        algorithm: BaseAlgorithm,
        priority: int = 8,
        **kwargs,
    ):
        super().__init__(jid, password, **kwargs)
        self.algorithm = algorithm
        self.priority = priority

    async def setup(self) -> None:
        config = self.algorithm.config

        logger.info(
            "Generic algorithm agent started: jid=%s, algorithm=%s",
            self.jid,
            config.name,
        )

        self.state_lock = asyncio.Lock()

        self.latest_records: dict[str, AlgorithmInputRecord] = {}
        self.updated_gateway_guids: set[str] = set()
        self.latest_outputs_by_key: dict[str, dict[str, Any]] = {}

        self.last_successful_run_monotonic: float | None = None

        self.output_file = Path(config.output_file) if config.output_file else None

        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.output_jid = config.output_jid

        self.set_agent_status(
            AgentStatus.ONLINE_IDLE,
            f"{config.name} waiting for input",
            priority=self.priority,
        )
        self.set_offline_message("stopped normally")

        self.add_behaviour(
            GenericAlgorithmBehaviour(),
            make_input_template(config),
        )

    async def handle_record(
        self,
        record: AlgorithmInputRecord,
        sender_behaviour: GenericAlgorithmBehaviour,
    ) -> None:
        config = self.algorithm.config

        if record.gateway_guid:
            self.latest_records[record.gateway_guid] = record
            self.updated_gateway_guids.add(record.gateway_guid)

        if config.mode == AlgorithmMode.PER_MESSAGE:
            await self.run_per_message(
                record=record,
                sender_behaviour=sender_behaviour,
            )
            return

        if config.mode == AlgorithmMode.COMPLETE_BATCH:
            await self.run_complete_batch(
                sender_behaviour=sender_behaviour,
            )
            return

        self.set_agent_status(
            AgentStatus.DEGRADED,
            "unsupported algorithm mode",
            priority=self.priority,
        )

    async def run_per_message(
        self,
        record: AlgorithmInputRecord,
        sender_behaviour: GenericAlgorithmBehaviour,
    ) -> None:
        try:
            if record.gateway_guid:
                records: Mapping[str, AlgorithmInputRecord] = {
                    record.gateway_guid: record
                }
            else:
                records = {
                    "_trigger": record
                }

            result = await self.algorithm.run(
                records=records,
                trigger_record=record,
            )

            await self.apply_algorithm_result(
                result=result,
                sender_behaviour=sender_behaviour,
            )

        except Exception:
            self.set_agent_status(
                AgentStatus.DEGRADED,
                f"{self.algorithm.config.name} error",
                priority=self.priority,
            )
            logger.exception(
                "Algorithm %s failed in PER_MESSAGE mode",
                self.algorithm.config.name,
            )

    async def run_complete_batch(
        self,
        sender_behaviour: GenericAlgorithmBehaviour,
    ) -> None:
        config = self.algorithm.config

        missing = [
            gateway_guid
            for gateway_guid in config.required_gateway_guids
            if gateway_guid not in self.latest_records
        ]

        if missing:
            self.set_agent_status(
                AgentStatus.ONLINE_IDLE,
                f"{config.name} waiting for {len(missing)} input(s)",
                priority=self.priority,
            )
            return

        if config.require_all_updated:
            not_updated = [
                gateway_guid
                for gateway_guid in config.required_gateway_guids
                if gateway_guid not in self.updated_gateway_guids
            ]

            if not_updated:
                self.set_agent_status(
                    AgentStatus.ONLINE_IDLE,
                    f"{config.name} waiting for complete batch",
                    priority=self.priority,
                )
                return

        stale = self.get_stale_gateway_guids()

        if stale:
            self.set_agent_status(
                AgentStatus.DEGRADED,
                f"{config.name} stale input data",
                priority=self.priority,
            )
            logger.warning(
                "Algorithm %s cannot run because data is stale: %s",
                config.name,
                stale,
            )
            return

        if config.require_same_timestamp:
            timestamps = {
                self.latest_records[gateway_guid].timestamp
                for gateway_guid in config.required_gateway_guids
            }

            if len(timestamps) > 1:
                self.set_agent_status(
                    AgentStatus.ONLINE_IDLE,
                    f"{config.name} waiting for synchronized data",
                    priority=self.priority,
                )
                logger.debug(
                    "Algorithm %s waits for synchronized timestamps: %s",
                    config.name,
                    sorted(str(item) for item in timestamps),
                )
                return

        try:
            records = {
                gateway_guid: self.latest_records[gateway_guid]
                for gateway_guid in config.required_gateway_guids
            }

            result = await self.algorithm.run(records=records)

            await self.apply_algorithm_result(
                result=result,
                sender_behaviour=sender_behaviour,
            )

            self.updated_gateway_guids.clear()

        except Exception:
            self.set_agent_status(
                AgentStatus.DEGRADED,
                f"{config.name} error",
                priority=self.priority,
            )
            logger.exception(
                "Algorithm %s failed in COMPLETE_BATCH mode",
                config.name,
            )

    def get_stale_gateway_guids(self) -> list[str]:
        config = self.algorithm.config
        now = time.monotonic()

        stale = []

        for gateway_guid in config.required_gateway_guids:
            record = self.latest_records.get(gateway_guid)

            if record is None:
                stale.append(gateway_guid)
                continue

            age = now - record.received_monotonic

            if age > config.max_data_age_seconds:
                stale.append(gateway_guid)

        return stale

    async def apply_algorithm_result(
        self,
        result: AlgorithmResult,
        sender_behaviour: GenericAlgorithmBehaviour,
    ) -> None:
        config = self.algorithm.config

        forwarding_ok = True

        if result.outputs:
            forwarding_ok = await self.forward_outputs(
                outputs=result.outputs,
                sender_behaviour=sender_behaviour,
            )
            self.write_outputs(result.outputs)

        if not forwarding_ok:
            self.set_agent_status(
                AgentStatus.DEGRADED,
                f"{config.name} output forwarding failed",
                priority=self.priority,
            )
            logger.warning(
                "Algorithm %s completed, but output forwarding failed",
                config.name,
            )
            return

        self.last_successful_run_monotonic = time.monotonic()

        self.set_agent_status(
            result.status,
            result.status_message,
            priority=self.priority,
        )
        self.set_offline_message(
            f"stopped normally; previous status {result.status.value}"
        )

        logger.info(
            "Algorithm %s completed: %s",
            config.name,
            result.status_message,
        )

    async def forward_outputs(
        self,
        outputs: list[dict[str, Any]],
        sender_behaviour: GenericAlgorithmBehaviour,
    ) -> bool:
        config = self.algorithm.config

        if not self.output_jid or not config.output_ontology:
            return True

        success = True

        for output in outputs:
            msg = Message(to=self.output_jid)
            msg.set_metadata("performative", "inform")
            msg.set_metadata("ontology", config.output_ontology)
            msg.body = json.dumps(output, ensure_ascii=False)

            try:
                await sender_behaviour.send(msg)

            except Exception:
                success = False
                logger.exception(
                    "Algorithm %s failed to forward output to %s",
                    config.name,
                    self.output_jid,
                )

        return success

    def write_outputs(self, outputs: list[dict[str, Any]]) -> None:
        config = self.algorithm.config

        if not self.output_file:
            return

        if config.output_storage_strategy == OutputStorageStrategy.NONE:
            return

        if config.output_storage_strategy == OutputStorageStrategy.LATEST:
            if len(outputs) == 1:
                payload: Any = outputs[0]
            else:
                payload = outputs

            self.atomic_write_json(payload)
            return

        if config.output_storage_strategy == OutputStorageStrategy.LATEST_BY_KEY:
            for output in outputs:
                key = output.get(config.output_key_field)

                if key is None:
                    logger.warning(
                        "Algorithm %s output has no key field %s",
                        config.name,
                        config.output_key_field,
                    )
                    continue

                self.latest_outputs_by_key[str(key)] = output

            payload = list(self.latest_outputs_by_key.values())
            payload.sort(
                key=lambda item: str(item.get(config.output_key_field, ""))
            )

            self.atomic_write_json(payload)
            return

        logger.warning(
            "Algorithm %s uses unsupported output storage strategy: %s",
            config.name,
            config.output_storage_strategy,
        )

    def atomic_write_json(self, payload: Any) -> None:
        if not self.output_file:
            return

        tmp_file = self.output_file.with_name(f".{self.output_file.name}.tmp")

        with tmp_file.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(tmp_file, self.output_file)