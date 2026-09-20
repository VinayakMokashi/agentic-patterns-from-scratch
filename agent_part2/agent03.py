import os
import sys

from colorama import Fore, Style
from dotenv import load_dotenv
from groq import Groq

# The shared toolkit lives in agent_part1; add it to the import path so part 2 reuses it
# instead of keeping a second copy that can drift out of date.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent_part1"))

from agent_pattern_utils import (  # noqa: E402
    FixedFirstChatHistory,
    build_prompt_structure,
    completions_create,
    resolve_model,
    update_chat_history,
)

load_dotenv()


class ReflectionAgent:
    def __init__(self, model: str | None = None):
        self.client = Groq()
        self.model = resolve_model(model)
        self.BASE_GENERATION_SYSTEM_PROMPT = """
        Your task is to Generate the best content possible for the user's request.
        If the user provides critique, respond with a revised version of your previous attempt.
        You must always output the revised content.
        """

        self.BASE_REFLECTION_SYSTEM_PROMPT = """
        You are tasked with generating critique and recommendations to the user's generated content.
        If the user content has something wrong or something to be improved, output a list of recommendations
        and critiques. If the user content is ok and there's nothing to change, output this: <OK>
        """

    def _request_completion( self, history: list, verbose: int = 0, log_title: str = "COMPLETION", log_color: str = "",):
        output = completions_create(self.client, history, self.model)

        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output, Style.RESET_ALL)

        return output

    def generate(self, generation_history: list, verbose: int = 0) -> str:
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.BLUE
        )

    def reflect(self, reflection_history: list, verbose: int = 0) -> str:
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )

    def run( self, user_msg: str, generation_system_prompt: str = "", reflection_system_prompt: str = "", n_steps: int = 10, verbose: int = 0,) -> tuple[str, int]:
        generation_system_prompt += self.BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += self.BASE_REFLECTION_SYSTEM_PROMPT

        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user"),
            ],
            total_length=3,
        )

        reflection_history = FixedFirstChatHistory(
            [build_prompt_structure(prompt=reflection_system_prompt, role="system")],
            total_length=3,
        )

        generation, step = "", -1

        for step in range(n_steps):
            generation = self.generate(generation_history, verbose=verbose)
            update_chat_history(generation_history, generation, "assistant")
            update_chat_history(reflection_history, generation, "user")

            critique = self.reflect(reflection_history, verbose=verbose)

            if "<OK>" in critique:
                print(
                    Fore.RED,
                    "\n\nStop Sequence found. Stopping the reflection loop ... \n\n",
                    Style.RESET_ALL,
                )
                break

            update_chat_history(generation_history, critique, "user")
            update_chat_history(reflection_history, critique, "assistant")

        return generation, step


if __name__ == "__main__":
    agent = ReflectionAgent()

    generation_system_prompt = "You are a Python programmer tasked with generating high quality Python code"

    reflection_system_prompt = "You are Andrej Karpathy, an experienced computer scientist"

    user_msg = "Generate a Python implementation of the Merge Sort algorithm"


    final_response, step = agent.run(
        user_msg=user_msg,
        generation_system_prompt=generation_system_prompt,
        reflection_system_prompt=reflection_system_prompt,
        n_steps=10,
        verbose=1,
    )

    print(final_response)
