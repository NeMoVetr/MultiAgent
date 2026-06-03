import asyncio
import logging
import os
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import spade
from dotenv import load_dotenv

from logging_config import configure_logging

load_dotenv()
configure_logging()

from DataQualityAgent import DataQualityAgent
from IrrigatorAgent import IrrigatorAgent
from SensorAgent import (
    OpticalRainGaugeAgent,
    SM9560BAgent,
    TBQ02CAgent,
    THPWNJAgent,
    TR4H01XAgent,
    XM8504Agent,
)

logger = logging.getLogger(__name__)

AGENT_HOST = os.getenv("AGENT_HOST")
XMPP_PORT = int(os.getenv("XMPP_PORT"))
AGENT_PASSWORD = os.getenv("SPADE_AGENT_PASSWORD")

DATA_QUALITY_AGENT_JID = f"data_quality@{AGENT_HOST}"

IRRIGATOR_AGENT_JID = f"irrigator@{AGENT_HOST}"

SENSOR_AGENT_DEFINITIONS = [
    ("sm9560b", SM9560BAgent),
    ("thpwnj", THPWNJAgent),
    ("tbq02c", TBQ02CAgent),
    ("xm8504", XM8504Agent),
    ("tr4h01x", TR4H01XAgent),
    ("optical_rain", OpticalRainGaugeAgent),
]


def create_data_quality_agent() -> DataQualityAgent:
    agent = DataQualityAgent(
        DATA_QUALITY_AGENT_JID,
        AGENT_PASSWORD,
        port=XMPP_PORT,
        verify_security=False,
        priority=7,
    )
    agent.set("irrigator_agent_jid", IRRIGATOR_AGENT_JID)
    return agent


def create_irrigator_agent() -> IrrigatorAgent:
    return IrrigatorAgent(
        IRRIGATOR_AGENT_JID,
        AGENT_PASSWORD,
        port=XMPP_PORT,
        verify_security=False,
        priority=8,
    )


def create_sensor_agents():
    agents = []

    for localpart, agent_class in SENSOR_AGENT_DEFINITIONS:
        jid = f"{localpart}@{AGENT_HOST}"
        agent = agent_class(
            jid,
            AGENT_PASSWORD,
            port=XMPP_PORT,
            verify_security=False,
        )
        agent.set("data_quality_agent_jid", DATA_QUALITY_AGENT_JID)
        agents.append(agent)

    return agents


async def main() -> None:
    irrigator_agent = create_irrigator_agent()
    data_quality_agent = create_data_quality_agent()
    sensor_agents = create_sensor_agents()

    started_agents = []

    try:
        logger.info("Starting irrigator agent: %s", irrigator_agent.jid)
        await irrigator_agent.start(auto_register=True)
        started_agents.append(irrigator_agent)
        logger.info("Irrigator agent started: %s", irrigator_agent.jid)

        logger.info("Starting data quality agent: %s", data_quality_agent.jid)
        await data_quality_agent.start(auto_register=True)
        started_agents.append(data_quality_agent)
        logger.info("Data quality agent started: %s", data_quality_agent.jid)

        await asyncio.sleep(1)

        for agent in sensor_agents:
            logger.info("Starting sensor agent: %s", agent.jid)
            await agent.start(auto_register=True)
            started_agents.append(agent)
            logger.info("Sensor agent started: %s", agent.jid)

        logger.info("All agents started")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")

    except asyncio.CancelledError:
        logger.info("Main task cancelled")
        raise

    except Exception:
        logger.exception("Application failed")

    finally:
        if started_agents:
            logger.info("Stopping agents")

            await asyncio.gather(
                *[
                    agent.stop()
                    for agent in reversed(started_agents)
                    if agent.is_alive()
                ],
                return_exceptions=True,
            )

            logger.info("Agents stopped")


if __name__ == "__main__":
    spade.run(main())
