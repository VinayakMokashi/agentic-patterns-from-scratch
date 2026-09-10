import json
import re

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agent_pattern_utils import *
import math

load_dotenv()


class ReactAgent:
    def __init__( self, tools, model="llama-3.3-70b-versatile", system_prompt=""):
        self.client = Groq()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools 
        self.tools_dict = {tool.name: tool for tool in self.tools}
        
        self.REACT_SYSTEM_PROMPT = """
            You operate by running a loop with the following steps: Thought, Action, Observation.
            You are provided with function signatures within <tools></tools> XML tags.
            You may call one or more functions to assist with the user query. Don' make assumptions about what values to plug
            into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

            For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

            <tool_call>
            {"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
            </tool_call>

            Here are the available tools / actions:

            <tools>
            %s
            </tools>

            Example session:

            <question>What's the current temperature in Madrid?</question>
            <thought>I need to get the current weather in Madrid</thought>
            <tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

            You will be called again with this:

            <observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

            You then output:

            <response>The current temperature in Madrid is 25 degrees Celsius</response>

            Additional constraints:

            - If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
        """

    def add_tool_signatures(self) -> str:
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")
            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")
            observations[validated_tool_call["id"]] = result

        return observations

    def run( self, user_msg: str, max_rounds: int = 10) -> str:
        user_prompt = build_prompt_structure(
            prompt=user_msg, role="user", tag="question"
        )
        self.system_prompt += (
            "\n" + self.REACT_SYSTEM_PROMPT % self.add_tool_signatures()
        )
        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system",
                ),
                user_prompt,
            ]
        )

        for iteration_index in range(max_rounds):
            completion = completions_create(self.client, chat_history, self.model)
            response = extract_tag_content(str(completion), "response")
            if response.found:
                return response.content[0]

            thought = extract_tag_content(str(completion), "thought")
            tool_calls = extract_tag_content(str(completion), "tool_call")

            update_chat_history(chat_history, completion, "assistant")

            print(Fore.MAGENTA + f"\nThought: {thought.content[0]}")

            if tool_calls.found:
                observations = self.process_tool_calls(tool_calls.content)
                print(Fore.BLUE + f"\nObservations: {observations}")
                update_chat_history(chat_history, f"{observations}", "user")

        return completions_create(self.client, chat_history, self.model)


@tool
def sum_two_elements(a: int, b: int) -> int:
    return a + b


@tool
def multiply_two_elements(a: int, b: int) -> int:
    return a * b


@tool
def compute_log(x: int) -> float | str:
    if x <= 0:
        return "Logarithm is undefined for values less than or equal to 0."
    return math.log(x)


if __name__ == "__main__":
    agent = ReactAgent(tools=[sum_two_elements, multiply_two_elements, compute_log])
    response = agent.run(user_msg="I want to calculate the sum of 1234 and 5678 and multiply the result by 5. Then, I want to take the logarithm of this result")