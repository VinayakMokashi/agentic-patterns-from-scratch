import json
import requests

from colorama import Fore, Style
from dotenv import load_dotenv
from groq import Groq

from agent_pattern_utils import *
load_dotenv()


class ToolAgent:
    def __init__( self, tools: Tool | list[Tool], model: str | None = None,) -> None:
        self.client = Groq()
        self.model = resolve_model(model)
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        self.TOOL_SYSTEM_PROMPT = """
        You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
        You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
        into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.
        For each function call return a json object with function name and arguments within <tool_call></tool_call>
        XML tags as follows:

        <tool_call>
        {"name": <function-name>,"arguments": <args-dict>,  "id": <monotonically-increasing-id>}
        </tool_call>

        Here are the available tools:

        <tools>
        %s
        </tools>

        Write the <tool_call></tool_call> tags directly in your reply text. Do not use native function calling.
        """

    def add_tool_signatures(self) -> str:
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        return run_tool_calls(tool_calls_content, self.tools_dict)

    def run(self, user_msg: str,) -> str:
        user_prompt = build_prompt_structure(prompt=user_msg, role="user")

        tool_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.TOOL_SYSTEM_PROMPT % self.add_tool_signatures(),
                    role="system",
                ),
                user_prompt,
            ]
        )
        agent_chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt="You are a helpful assistant. Observations contain real, up-to-date results "
                    "from tools that were just run for the user's request. Base your answer on them. "
                    "Respond in plain text only. Do not call any tools.",
                    role="system",
                ),
                user_prompt,
            ]
        )

        tool_call_response = completions_create(
            self.client, messages=tool_chat_history, model=self.model
        )
        tool_calls = extract_tag_content(str(tool_call_response), "tool_call")

        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(
                agent_chat_history,
                "Observation (results of the tools run for this request):\n"
                f"Tool calls: {tool_calls.content}\n"
                f"Results by call id: {observations}\n"
                "Use this observation to answer the request.",
                "user",
            )

        return completions_create(self.client, agent_chat_history, self.model)


@tool
def fetch_top_hacker_news_stories(top_n: int):
    """Fetch the titles and URLs of the current top `top_n` stories on Hacker News."""
    top_stories_url = 'https://hacker-news.firebaseio.com/v0/topstories.json'

    try:
        response = requests.get(top_stories_url, timeout=10)
        response.raise_for_status()

        top_story_ids = response.json()[:top_n]

        top_stories = []

        for story_id in top_story_ids:
            story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
            story_response = requests.get(story_url, timeout=10)
            story_response.raise_for_status()  # Check for HTTP errors
            story_data = story_response.json()

            top_stories.append({
                'title': story_data.get('title', 'No title'),
                'url': story_data.get('url', 'No URL available'),
            })

        return json.dumps(top_stories)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    hn_tool = fetch_top_hacker_news_stories
    tool_agent = ToolAgent(tools=[hn_tool])

    output_1 = tool_agent.run(user_msg="Tell me your name")
    print(Fore.YELLOW + f"\nResponse 1:\n{output_1}" + Style.RESET_ALL)

    output_2 = tool_agent.run(user_msg="Tell me the top 5 Hacker News stories right now")
    print(Fore.YELLOW + f"\nResponse 2:\n{output_2}" + Style.RESET_ALL)
