import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import os

import spade
from dotenv import load_dotenv

from logging_config import configure_logging

load_dotenv()
configure_logging()

from IntegrationAgent import GenericAlgorithmAgent
from SensorAgent import SM7005Agent, SM9560BAgent, THPWNJAgent, TBQ02CAgent, SensorManagerAgent
from algorithms import TimeNormalizationAlgorithm, IrrigationAlgorithm

logger = logging.getLogger(__name__)

XMPP_HOST = os.getenv("XMPP_HOST", "localhost")
XMPP_PORT = int(os.getenv("XMPP_PORT", "5222"))
AGENT_PASSWORD = os.getenv("SPADE_AGENT_PASSWORD", "agent_password_123")

MANAGER_JID = os.getenv("MANAGER_JID", f"manager@{XMPP_HOST}")
NORMALIZER_AGENT_JID = os.getenv("NORMALIZER_AGENT_JID", f"normalizer@{XMPP_HOST}")
ALGORITHM_AGENT_JID = os.getenv("ALGORITHM_AGENT_JID", f"irrigation@{XMPP_HOST}")

SENSOR_AGENT_DEFINITIONS = [
    ("sm9560b", SM9560BAgent),
    ("thpwnj", THPWNJAgent),
    ("tbq02c", TBQ02CAgent),
    ("sm7005", SM7005Agent),
]


def create_normalizer_agent() -> GenericAlgorithmAgent:
    return GenericAlgorithmAgent(
        NORMALIZER_AGENT_JID,
        AGENT_PASSWORD,
        algorithm=TimeNormalizationAlgorithm.from_env(),
        port=XMPP_PORT,
        verify_security=False,
        priority=7,
    )


def create_irrigation_agent() -> GenericAlgorithmAgent:
    return GenericAlgorithmAgent(
        ALGORITHM_AGENT_JID,
        AGENT_PASSWORD,
        algorithm=IrrigationAlgorithm.from_env(),
        port=XMPP_PORT,
        verify_security=False,
        priority=8,
    )

def create_manager_agent() -> SensorManagerAgent:
    return SensorManagerAgent(
        MANAGER_JID,
        AGENT_PASSWORD,
        port=XMPP_PORT,
        verify_security=False,
    )


def create_sensor_agents():
    agents = []

    for localpart, agent_class in SENSOR_AGENT_DEFINITIONS:
        jid = f"{localpart}@{XMPP_HOST}"

        agent = agent_class(
            jid,
            AGENT_PASSWORD,
            port=XMPP_PORT,
            verify_security=False,
        )

        agent.set("manager_jid", MANAGER_JID)
        agents.append(agent)

    return agents


async def main() -> None:
    irrigation_agent = create_irrigation_agent()
    normalizer_agent = create_normalizer_agent()
    manager_agent = create_manager_agent()
    sensor_agents = create_sensor_agents()

    started_agents = []

    try:
        logger.info("Starting irrigation algorithm agent: %s", irrigation_agent.jid)
        await irrigation_agent.start(auto_register=True)
        started_agents.append(irrigation_agent)
        logger.info("Irrigation algorithm agent started: %s", irrigation_agent.jid)

        logger.info("Starting time normalizer agent: %s", normalizer_agent.jid)
        await normalizer_agent.start(auto_register=True)
        started_agents.append(normalizer_agent)
        logger.info("Time normalizer agent started: %s", normalizer_agent.jid)

        logger.info("Starting manager agent: %s", manager_agent.jid)
        await manager_agent.start(auto_register=True)
        started_agents.append(manager_agent)
        logger.info("Manager agent started: %s", manager_agent.jid)
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