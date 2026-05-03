import asyncio
import spade
from dotenv import load_dotenv
import os

from ManagingАgents import SensorManagerAgent
from SensorAgent import SM7005Agent, SM9560BAgent, THPWNJAgent, TBQ02CAgent

load_dotenv()

XMPP_HOST = os.getenv("XMPP_HOST", "localhost")
XMPP_PORT = int(os.getenv("XMPP_PORT", "5222"))
AGENT_PASSWORD = os.getenv("SPADE_AGENT_PASSWORD", "agent_password_123")

MANAGER_JID = os.getenv("MANAGER_JID", f"manager@{XMPP_HOST}")

AGENT_DEFINITIONS = [
    ("sm9560b", SM9560BAgent),
    ("thpwnj", THPWNJAgent),
    ("tbq02c", TBQ02CAgent),
    ("sm7005", SM7005Agent),
]



def create_manager_agent():
    return SensorManagerAgent(
        MANAGER_JID,
        AGENT_PASSWORD,
        port=XMPP_PORT,
        verify_security=False,
    )


def create_sensor_agents():
    agents = []

    for localpart, agent_class in AGENT_DEFINITIONS:
        jid = f"{localpart}@{XMPP_HOST}"

        agent = agent_class(
            jid,
            AGENT_PASSWORD,
            port=XMPP_PORT,
        )

        agents.append(agent)

    return agents


async def main():
    #manager_agent = create_manager_agent()
    sensor_agents = create_sensor_agents()

    started_agents = []

    try:
        # print(f"Starting manager agent {manager_agent.jid}...")
        # await manager_agent.start(auto_register=True)
        # started_agents.append(manager_agent)
        # print(f"✅ Manager agent started: {manager_agent.jid}")

        await asyncio.sleep(1)

        for agent in sensor_agents:
            print(f"Starting sensor agent {agent.jid}...")
            await agent.start(auto_register=True)
            started_agents.append(agent)
            print(f"✅ Sensor agent started: {agent.jid}")

        print("✅ All agents started.")
        print("Press Ctrl+C to stop.")

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Stopping by Ctrl+C...")

    except Exception as e:
        print(f"❌ Error while starting or running agents: {e}")

    finally:
        if started_agents:
            print("Stopping agents...")

            await asyncio.gather(
                *[
                    agent.stop()
                    for agent in reversed(started_agents)
                    if agent.is_alive()
                ],
                return_exceptions=True,
            )

            print("✅ Agents stopped.")


if __name__ == "__main__":
    spade.run(main())
