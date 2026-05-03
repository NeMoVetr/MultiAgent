import asyncio
from datetime import datetime, timezone
from enum import Enum

from spade.agent import Agent
from spade.presence import PresenceShow, PresenceType


class AgentStatus(str, Enum):
    WORKING = "WORKING"
    ONLINE_IDLE = "ONLINE_IDLE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def presence_show_for_status(status: AgentStatus) -> PresenceShow:
    if status == AgentStatus.WORKING:
        return PresenceShow.CHAT

    if status == AgentStatus.ONLINE_IDLE:
        return PresenceShow.AWAY

    if status == AgentStatus.DEGRADED:
        return PresenceShow.DND

    return PresenceShow.NONE


class StatusAwareAgent(Agent):
    """
    Базовый SPADE-агент с управляемым XMPP Presence.

    WORKING      -> show=chat
    ONLINE_IDLE  -> show=away
    DEGRADED     -> show=dnd
    OFFLINE      -> unavailable
    """

    def set_agent_status(
        self,
        status: AgentStatus | str,
        message: str,
        priority: int = 5,
    ) -> None:
        status = AgentStatus(status)

        clean_message = " ".join(str(message).split())
        status_text = f"{status.value}: {clean_message}"[:250]

        self.set("last_agent_status", status.value)
        self.set("last_status_text", status_text)

        if not self.presence:
            return

        if status == AgentStatus.OFFLINE:
            self.presence.set_presence(
                presence_type=PresenceType.UNAVAILABLE,
                show=PresenceShow.NONE,
                status=status_text,
                priority=0,
            )
            return

        self.presence.set_presence(
            presence_type=PresenceType.AVAILABLE,
            show=presence_show_for_status(status),
            status=status_text,
            priority=priority,
        )

    def set_offline_message(self, message: str) -> None:
        self.set("offline_message", " ".join(str(message).split())[:220])

    async def _async_stop(self) -> None:
        last_status = self.get("last_agent_status") or AgentStatus.ONLINE_IDLE.value
        offline_message = self.get("offline_message")

        if not offline_message:
            offline_message = f"stopped normally; previous status {last_status}"

        if self.presence:
            self.set_agent_status(
                AgentStatus.OFFLINE,
                offline_message,
                priority=0,
            )

            await asyncio.sleep(0.3)

        for behaviour in list(self.behaviours):
            behaviour.kill()

        if self.web.is_started():
            await self.web.runner.cleanup()

        if self.is_alive():
            await self.client.disconnect()

        self._alive.clear()