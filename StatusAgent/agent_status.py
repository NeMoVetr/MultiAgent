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
    def set_agent_status(
        self,
        status: AgentStatus | str,
        detail: str = "",
        priority: int = 5,
    ) -> None:
        status = AgentStatus(status)

        if detail:
            text = f"{status.value}: {detail}"[:250]
        else:
            text = status.value

        self.set("last_agent_status", status.value)
        self.set("last_status_text", text)

        if not self.presence:
            return

        if status == AgentStatus.OFFLINE:
            self.presence.set_presence(
                presence_type=PresenceType.UNAVAILABLE,
                show=PresenceShow.NONE,
                status=text,
                priority=0,
            )
            return

        self.presence.set_presence(
            presence_type=PresenceType.AVAILABLE,
            show=presence_show_for_status(status),
            status=text,
            priority=priority,
        )

    def set_offline_detail(self, detail: str) -> None:
        self.set("offline_detail", detail[:220])

    async def _async_stop(self) -> None:
        offline_detail = self.get("offline_detail")

        if not offline_detail:
            offline_detail = f"stopped at {utc_now_iso()}"

        if self.presence:
            self.set_agent_status(
                AgentStatus.OFFLINE,
                offline_detail,
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