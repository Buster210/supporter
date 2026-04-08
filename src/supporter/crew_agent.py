import asyncio
from typing import Any
from crewai import Agent, Task, Crew, Process
from .crew_adapter import SupporterLLM
from .index import LLMFactory
from .logger import logger


class CrewManager:
    def __init__(self, status_callback: Any = None):
        provider = LLMFactory.get_provider()
        self.llm = SupporterLLM(provider=provider, status_callback=status_callback)

    def _assemble_research_crew(self, topic: str) -> Crew:
        researcher = Agent(
            role="Senior Research Analyst",
            goal="Uncover cutting-edge developments and provide deep insights on {topic}",
            backstory="You are a veteran researcher with an eye for detail. \n            You excel at finding non-obvious connections and trends.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
        writer = Agent(
            role="Technical Content Strategist",
            goal="Synthesize complex information into clear, actionable, and engaging reports",
            backstory="You are an expert communicator who can take technical jargon \n            and turn it into a narrative that humans actually want to read.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
        research_task = Task(
            description=f"Conduct a comprehensive research on: {topic}. Focus on accuracy and depth.",
            expected_output="A detailed bulleted list of key findings and supporting data.",
            agent=researcher,
        )
        write_task = Task(
            description=f"Synthesize the research findings into a coherent report for: {topic}",
            expected_output="Professional markdown report addressing the research objective.",
            agent=writer,
            context=[research_task],
        )
        return Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True,
        )

    async def coordinate_execution(self, prompt: str) -> tuple[str, list[str]]:
        try:
            crew = self._assemble_research_crew(prompt)
            result = await asyncio.to_thread(crew.kickoff, inputs={"topic": prompt})
            agent_roles = []
            if hasattr(result, "tasks_output"):
                agent_roles = [
                    task.agent for task in result.tasks_output if hasattr(task, "agent")
                ]
            if not agent_roles:
                agent_roles = [a.role for a in crew.agents]
            return (str(result), list(set(agent_roles)))
        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
            return (f"Error executing crew: {e}", [])
