import asyncio
from typing import Any

from crewai import Agent, Crew, Process, Task

from ..config import RESEARCHER_ROLE, WRITER_ROLE
from ..llm_types import LLMResult
from ..logger import logger
from .crew_adapter import SupporterLLM


class CrewManager:
    def __init__(self, provider: Any, status_callback: Any = None):
        logger.debug("Initializing CrewManager")
        self.llm = SupporterLLM(provider=provider, status_callback=status_callback)

    def _assemble_research_crew(self, topic: str) -> Crew:
        logger.debug(f"Entering _assemble_research_crew (topic: {topic})")

        researcher = Agent(
            role=RESEARCHER_ROLE,
            goal=(
                "Uncover cutting-edge developments and provide deep insights on {topic}"
            ),
            backstory=(
                "You are a veteran researcher with an eye for detail. "
                "You excel at finding non-obvious connections and trends."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        writer = Agent(
            role=WRITER_ROLE,
            goal=(
                "Synthesize complex information into clear, actionable, "
                "and engaging reports"
            ),
            backstory=(
                "You are an expert communicator who can take technical jargon "
                "and turn it into a narrative that humans actually want to read."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        research_task = Task(
            description=(
                f"Conduct a comprehensive research on: {topic}. "
                "Focus on accuracy and depth."
            ),
            expected_output=(
                "A detailed bulleted list of key findings and supporting data."
            ),
            agent=researcher,
        )

        write_task = Task(
            description=(
                f"Synthesize the research findings into a coherent report for: {topic}"
            ),
            expected_output=(
                "Professional markdown report addressing the research objective."
            ),
            agent=writer,
            context=[research_task],
        )

        return Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True,
        )

    async def coordinate_execution(self, prompt: str) -> LLMResult:
        logger.debug(f"Entering coordinate_execution (prompt: {prompt[:50]}...)")

        try:
            crew = self._assemble_research_crew(prompt)
            result = await asyncio.to_thread(crew.kickoff, inputs={"topic": prompt})
            agent_roles = []
            if hasattr(result, "tasks_output"):
                agent_roles = [
                    task.agent for task in result.tasks_output if hasattr(task, "agent")
                ]

            if not agent_roles:
                agent_roles = [agent.role for agent in crew.agents]

            return LLMResult(
                text=str(result),
                model="CrewAI (Multi-Agent)",
                usage={"agents": list(set(agent_roles))},
            )

        except Exception as error:
            logger.error(f"Crew orchestration failed: {error}")
            return LLMResult(text=f"Error executing crew: {error}")
        finally:
            logger.debug("Exiting coordinate_execution")
