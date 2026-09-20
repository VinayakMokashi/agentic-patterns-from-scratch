import os
import sys
from textwrap import dedent
from collections import deque

from colorama import Fore
from dotenv import load_dotenv

# The shared toolkit and the ReAct agent live in agent_part1; add it to the import path so
# part 2 reuses them instead of keeping a second copy that can drift out of date.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_part1"))

from agent_pattern_utils import Tool, fancy_print, tool  # noqa: E402
from agent02 import ReactAgent  # noqa: E402

load_dotenv()


class Agent:
    def __init__( self, name: str, backstory: str, task_description: str, task_expected_output: str = "", tools: list[Tool] | None = None, llm: str | None = None,):
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools or []
        )

        self.dependencies: list[Agent] = []  # Agents that this agent depends on
        self.dependents: list[Agent] = []  # Agents that depend on this agent

        self.context = ""

        Crew.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        self.add_dependent(other)
        return other  # Allow chaining

    def __lshift__(self, other):
        self.add_dependency(other)
        return other  # Allow chaining

    def __rrshift__(self, other):
        self.add_dependency(other)
        return self  # Allow chaining

    def __rlshift__(self, other):
        self.add_dependent(other)
        return self  # Allow chaining

    def add_dependency(self, other):
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, input_data):
        self.context += f"{self.name} received context: \n{input_data}"

    def create_prompt(self):
        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {self.context}
        </context>

        Your response:
        """
        ).strip()

        return prompt

    def run(self):
        msg = self.create_prompt()
        output = self.react_agent.run(user_msg=msg)

        for dependent in self.dependents:
            dependent.receive_context(output)
        return output


class Crew:
    current_crew = None

    def __init__(self):
        self.agents = []

    def __enter__(self):
        Crew.current_crew = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Crew.current_crew = None

    def add_agent(self, agent):
        self.agents.append(agent)

    @staticmethod
    def register_agent(agent):
        if Crew.current_crew is not None:
            Crew.current_crew.add_agent(agent)

    def topological_sort(self):
        in_degree = {agent: len(agent.dependencies) for agent in self.agents}
        queue = deque([agent for agent in self.agents if in_degree[agent] == 0])

        sorted_agents = []

        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)

            for dependent in current_agent.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_agents) != len(self.agents):
            raise ValueError(
                "Circular dependencies detected among agents, preventing a valid topological sort"
            )

        return sorted_agents

    def plot(self):
        """Render the dependency graph. Requires the optional `graphviz` package."""
        from graphviz import Digraph  # type: ignore  # imported lazily: optional dependency

        dot = Digraph(format="png")  # Set format to PNG for inline display

        for agent in self.agents:
            dot.node(agent.name)
            for dependency in agent.dependencies:
                dot.edge(dependency.name, agent.name)
        return dot

    def run(self):
        sorted_agents = self.topological_sort()
        for agent in sorted_agents:
            fancy_print(f"RUNNING AGENT: {agent}")
            print(Fore.RED + f"{agent.run()}")


@tool
def write_str_to_txt(string_data: str, txt_filename: str):
    """Write string_data to txt_filename as UTF-8 text."""
    with open(txt_filename, mode="w", encoding="utf-8") as file:
        file.write(string_data)

    print(f"Data successfully written to {txt_filename}")
    # Return a result, otherwise the observation the model receives is just "None".
    return f"Wrote {len(string_data)} characters to {txt_filename}"


if __name__ == "__main__":


    with Crew() as crew:
        agent_1 = Agent(
            name="Poet Agent",
            backstory="You are a huge Taylor Swift fan, who enjoys writing songs like her. Your IQ is 75",
            task_description="Write a poem about the meaning of life",
            task_expected_output="Just output the poem, without any title or introductory sentences",
        )

        agent_2 = Agent(
            name="Poem Translator Agent",
            backstory="You are an expert translator especially skilled in Hindi",
            task_description="Translate a poem into Hindi", 
            task_expected_output="Just output the translated poem and nothing else"
        )

        agent_3 = Agent(
            name="Writer Agent",
            backstory="You are an expert transcriber, that loves writing poems into txt files",
            task_description="You'll receive a Hindi poem in your context. You need to write the poem into './poem_hindi.txt' file",
            task_expected_output="A txt file containing the Hindi poem received from the context",
            tools=[write_str_to_txt],
        )

        agent_1 >> agent_2 >> agent_3

    # crew.plot() returns a graphviz Digraph of the dependency graph.
    # It needs the optional `graphviz` package, so it is not called here.
    crew.run()