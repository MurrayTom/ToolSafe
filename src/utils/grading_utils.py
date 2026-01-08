from typing import Any, List
import re
import json

from utils.tool_parser import *
# from inspect_ai.log import transcript
# from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, Model


def semantic_judge(prompt: str, message: str, model: object) -> str:
    if message in ["", {}, None]:
        return "fail"
    input_message = f"{prompt} {message}"
    output = model.generate(input_message)
    response = output
    response = response.strip().lower() if response else ""
    if response not in ["pass", "fail"]:
        print(
            f"Warning: semantic judge returned an invalid response which is not 'pass' or 'fail': '{response}'."
        )
    if isinstance(response, str):
        return response
    else:
        raise ValueError("Response is not a string")


def is_function_called(messages: List, function_name: str, agent_type: str) -> bool:
    """Returns True if the function named `function_name` is called by the model."""
    for message in messages:
        if message["role"] == "assistant":
            tool_name, _ = tool_extractor[agent_type](message["content"])
            if tool_name == function_name:
                return True
    return False


def times_function_called(agent_type: str, messages: List, function_name: str) -> int:
    """Returns the number of times the function named `function_name` is called by the model."""
    n = 0
    for message in messages:
        if message["role"] == "assistant":
            tool_name, _ = tool_extractor[agent_type](message["content"])
            if tool_name == function_name:
                n += 1
    return n


def get_all_function_arguments(agent_type: str, messages: List) -> List[Any]:
    """Returns a list of all the arguments of all the function calls in `messages`."""
    function_arguments = []
    for message in messages:
        if message["role"] == "assistant":
            tool_name, tool_arguments = tool_extractor[agent_type](message["content"])
            if tool_name:
                function_arguments.append(tool_arguments)
    return function_arguments


def get_all_function_arguments_of_function(
    agent_type: str, messages: List, function_name: str
) -> List[Any]:
    """Returns a list of all the arguments of the function named `function_name` in `messages`."""
    function_arguments = []
    for message in messages:
        if message["role"] == "assistant":
            tool_name, tool_arguments = tool_extractor[agent_type](message["content"])
            if tool_name == function_name:
                function_arguments.append(tool_arguments)
    return function_arguments


def get_args_of_function_call(agent_type: str, messages: List, function_name: str) -> Any:
    """Returns the *last* occurrence of a function named `function_name` in `messages`."""
    for message in messages[::-1]:
        if message["role"] == "assistant":
            tool_name, tool_arguments = tool_extractor[agent_type](message["content"])
            if tool_name == function_name:
                return tool_arguments

    # if no function named `function_name` is present in `messages`, return an empty dict
    return {}


def get_response_of_function_call(
    agent_type: str, messages: List, function_name: str
) -> Any:
    """Returns the *first* occurrence of a function named `function_name` in `messages`."""
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant":
            tool_name, tool_arguments = tool_extractor[agent_type](messages[i]["content"])
            if tool_name == function_name and i < len(messages)-1:
                if messages[i+1]["role"] == "user":
                    return messages[i+1]["content"]
    return ""


def get_all_function_calls(messages: List, agent_type: str) -> List[str]:
    """Returns a list of all the function calls in `messages`."""
    function_calls = []
    for message in messages:
        if message["role"] == "assistant":
            tool_name, _ = tool_extractor[agent_type](message["content"])
            function_calls.append(tool_name)
    return function_calls


def extract_first_text_message(messages: List) -> str:
    """Returns the first text message in `messages`."""
    for message in messages:
        if message["role"] == "assistant" and message["content"] is not None:
            return message["content"]
    return ""


def extract_last_text_message(messages: List) -> str:
    """Returns the last text message in `messages`."""
    for message in messages[::-1]:
        if message["role"] == "assistant" and message["content"] is not None:
            return message["content"]
    return ""


def is_correct_order(target_functions: List[str], messages: List, agent_type: str) -> bool:
    """Returns True if the target functions are called in the correct order."""
    all_function_calls = get_all_function_calls(messages, agent_type)

    # the model might call some irrelevant tools as well, but that can still be fine
    all_function_calls_except_irrelevant = [
        f for f in all_function_calls if f in target_functions
    ]

    # check if target_functions forms a subsequence of all_function_calls_except_irrelevant
    def is_subsequence(subseq: List[str], seq: List[str]) -> bool:
        it = iter(seq)
        return all(any(x == y for y in it) for x in subseq)

    return is_subsequence(target_functions, all_function_calls_except_irrelevant)
