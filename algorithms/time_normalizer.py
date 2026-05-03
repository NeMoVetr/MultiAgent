import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from StatusAgent import AgentStatus
from .algorithm_contract import (
    AlgorithmConfig,
    AlgorithmInputRecord,
    AlgorithmMode,
    AlgorithmResult,
    BaseAlgorithm,
    OutputStorageStrategy,
)

logger = logging.getLogger(__name__)


class TimestampNormalizationError(ValueError):
    pass


def trim_fraction_to_microseconds(value: str) -> str:
    """
    Python datetime поддерживает максимум 6 знаков микросекунд.

    Пример:
    2026-05-03T21:11:54.401123456Z
    ->
    2026-05-03T21:11:54.401123Z
    """
    return re.sub(r"(\.\d{6})\d+", r"\1", value)


def resolve_timezone(source_timezone: str):
    if source_timezone.upper() == "UTC":
        return timezone.utc

    try:
        return ZoneInfo(source_timezone)
    except ZoneInfoNotFoundError as error:
        raise TimestampNormalizationError(
            f"Unknown source timezone: {source_timezone}"
        ) from error


def parse_datetime_to_utc(
    timestamp: str | datetime,
    source_timezone: str = "UTC",
) -> datetime:
    """
    Переводит timestamp в timezone-aware datetime в UTC.

    Поддерживает:
    - 2026-05-03T21:11:54.401Z
    - 2026-05-03T23:11:54.401+02:00
    - 2026-05-03T21:11:54
    - datetime object

    Если timestamp без timezone, он считается временем в source_timezone.
    """

    if isinstance(timestamp, datetime):
        dt = timestamp
    else:
        if timestamp is None:
            raise TimestampNormalizationError("Timestamp is None")

        raw = str(timestamp).strip()

        if not raw:
            raise TimestampNormalizationError("Timestamp is empty")

        normalized = raw.replace(" ", "T").replace(",", ".")

        if normalized.endswith("Z") or normalized.endswith("z"):
            normalized = normalized[:-1] + "+00:00"

        normalized = trim_fraction_to_microseconds(normalized)

        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise TimestampNormalizationError(
                f"Invalid ISO timestamp: {timestamp}"
            ) from error

    if dt.tzinfo is None:
        source_tz = resolve_timezone(source_timezone)
        dt = dt.replace(tzinfo=source_tz)

    return dt.astimezone(timezone.utc)


def normalize_timestamp_to_minute_utc(
    timestamp: str | datetime,
    source_timezone: str = "UTC",
) -> str:
    """
    Зануляет секунды и микросекунды, переводит время в UTC.

    Пример:
    2026-05-03T21:11:54.401Z
    ->
    2026-05-03T21:11:00Z
    """

    dt_utc = parse_datetime_to_utc(
        timestamp=timestamp,
        source_timezone=source_timezone,
    )

    dt_utc = dt_utc.replace(second=0, microsecond=0)

    return dt_utc.strftime("%Y-%m-%dT%H:%M:00Z")


def normalize_gateway_record(
    record: dict[str, Any],
    source_timezone: str = "UTC",
) -> dict[str, Any]:
    """
    Нормализует запись формата:

    {
        "gateway_guid": "...",
        "timestamp": "...",
        "value": ...
    }

    Возвращает такую же структуру, но timestamp будет UTC с минутной гранулярностью.
    """

    gateway_guid = record.get("gateway_guid")
    timestamp = record.get("timestamp")

    if not gateway_guid:
        raise TimestampNormalizationError("gateway_guid is missing")

    if "value" not in record:
        raise TimestampNormalizationError("value is missing")

    normalized_timestamp = normalize_timestamp_to_minute_utc(
        timestamp=timestamp,
        source_timezone=source_timezone,
    )

    return {
        "gateway_guid": str(gateway_guid),
        "timestamp": normalized_timestamp,
        "value": record["value"],
    }


class TimeNormalizationAlgorithm(BaseAlgorithm):
    """
    Алгоритм нормализации времени.

    Он принимает одну запись:

    {
        "gateway_guid": "...",
        "timestamp": "2026-05-03T21:11:54.401Z",
        "value": ...
    }

    И возвращает:

    {
        "gateway_guid": "...",
        "timestamp": "2026-05-03T21:11:00Z",
        "value": ...
    }
    """

    def __init__(
        self,
        source_timezone: str = "UTC",
        output_jid: str | None = None,
        output_file: str | None = None,
    ):
        self.source_timezone = source_timezone

        self.config = AlgorithmConfig(
            name="time-normalization",
            mode=AlgorithmMode.PER_MESSAGE,
            input_ontology="normalization-input",
            output_ontology="algorithm-input",
            output_jid=output_jid,
            output_file=output_file,
            output_storage_strategy=OutputStorageStrategy.LATEST_BY_KEY,
            output_key_field="gateway_guid",
        )

    @classmethod
    def from_env(cls) -> "TimeNormalizationAlgorithm":
        return cls(
            source_timezone=os.getenv("SOURCE_TIMESTAMP_TIMEZONE", "UTC"),
            output_jid=os.getenv("ALGORITHM_AGENT_JID", "irrigation@localhost"),
            output_file=os.getenv(
                "NORMALIZED_SENSOR_STATE_FILE",
                "./data/sensors_state.json",
            ),
        )

    async def run(
        self,
        records: Mapping[str, AlgorithmInputRecord],
        trigger_record: AlgorithmInputRecord | None = None,
    ) -> AlgorithmResult:
        if trigger_record is None:
            raise ValueError("TimeNormalizationAlgorithm requires trigger_record")

        normalized = normalize_gateway_record(
            record=trigger_record.raw,
            source_timezone=self.source_timezone,
        )

        logger.debug(
            "Timestamp normalized for gateway %s: %s",
            normalized["gateway_guid"],
            normalized["timestamp"],
        )

        return AlgorithmResult(
            outputs=[normalized],
            status=AgentStatus.WORKING,
            status_message="timestamps normalized",
        )