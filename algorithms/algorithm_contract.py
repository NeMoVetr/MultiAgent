import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from StatusAgent import AgentStatus


class AlgorithmMode(str, Enum):
    PER_MESSAGE = "PER_MESSAGE"
    COMPLETE_BATCH = "COMPLETE_BATCH"


class OutputStorageStrategy(str, Enum):
    NONE = "NONE"
    LATEST = "LATEST"
    LATEST_BY_KEY = "LATEST_BY_KEY"


@dataclass(frozen=True)
class AlgorithmInputRecord:
    gateway_guid: str | None
    timestamp: str | None
    value: Any
    raw: dict[str, Any]
    received_monotonic: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class AlgorithmConfig:
    name: str
    mode: AlgorithmMode

    input_ontology: str
    output_ontology: str | None = None
    output_jid: str | None = None

    required_gateway_guids: tuple[str, ...] = ()

    require_all_updated: bool = True
    require_same_timestamp: bool = False
    max_data_age_seconds: float = 120.0

    output_file: str | None = None
    output_storage_strategy: OutputStorageStrategy = OutputStorageStrategy.NONE
    output_key_field: str = "gateway_guid"


@dataclass
class AlgorithmResult:
    outputs: list[dict[str, Any]] = field(default_factory=list)
    status: AgentStatus = AgentStatus.WORKING
    status_message: str = "algorithm completed"


class BaseAlgorithm(ABC):
    """
    Любой внешний алгоритм должен реализовать этот контракт.

    Агент сам:
    - принимает XMPP-сообщения;
    - собирает batch, если нужно;
    - проверяет полноту данных;
    - вызывает run(...);
    - пересылает outputs следующему агенту;
    - сохраняет outputs в файл.
    """

    config: AlgorithmConfig

    @abstractmethod
    async def run(
        self,
        records: Mapping[str, AlgorithmInputRecord],
        trigger_record: AlgorithmInputRecord | None = None,
    ) -> AlgorithmResult:
        pass