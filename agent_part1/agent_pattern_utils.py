import json
from typing import Callable
import re
from dataclasses import dataclass
import time

from colorama import Fore
from colorama import Style


def completions_create(client, messages: list, model: str) -> str:
    response = client.chat.completions.create(messages=messages, model=model)
    return str(response.choices[0].message.content)


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}


def update_chat_history(history: list, msg: str, role: str):
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    def __init__(self, messages=None, total_length=-1):
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages=None, total_length=-1):
        super().__init__(messages, total_length)

    def append(self, msg: str):
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    properties = tool_signature["parameters"]["properties"]

    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
    }

    for arg_name, arg_value in tool_call["arguments"].items():
        expected_type = properties[arg_name].get("type")

        if not isinstance(arg_value, type_mapping[expected_type]):
            tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

    return tool_call


class Tool:
    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature

    def __str__(self):
        return self.fn_signature

    def run(self, **kwargs):
        return self.fn(**kwargs)


def get_fn_signature(fn: Callable) -> dict:
    fn_signature: dict = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {"properties": {}},
    }
    schema = {
        k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"
    }
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


def tool(fn: Callable):
    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=fn_signature.get("name"), fn=fn, fn_signature=json.dumps(fn_signature)
        )

    return wrapper()


@dataclass
class TagContentResult:
    content: list[str]
    found: bool


def extract_tag_content(text, tag):
    tag_pattern = rf"<{tag}>(.*?)</{tag}>"
    matched_contents = re.findall(tag_pattern, text, re.DOTALL)

    result = TagContentResult(
        content=[content.strip() for content in matched_contents],
        found=bool(matched_contents),
    )
    return result


def fancy_print(message: str) -> None:
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)


def fancy_step_tracker(step: int, total_steps: int) -> None:
    fancy_print(f"STEP {step + 1}/{total_steps}")
