import re
import json
import sys

from copy import deepcopy
from types import FunctionType
from tqdm import tqdm
from agent.agent import *
from agent.agent_prompts import *

from utils.tool_parser import *

from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from pydantic import BaseModel
from typing_extensions import Self

from task_executor.agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from task_executor.agentdojo.agent_pipeline.errors import AbortAgentError
from task_executor.agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from task_executor.agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from task_executor.agentdojo.functions_runtime import Function, FunctionCall, FunctionsRuntime, TaskEnvironment
from task_executor.agentdojo.types import ChatMessage, MessageContentBlock, get_text_content_as_str
from task_executor.agentdojo.yaml_loader import ImportLoader

Env = TypeVar("Env", bound=TaskEnvironment)

def model_output_from_messages(
        messages: Sequence[ChatMessage],
    ) -> list[MessageContentBlock] | None:
        if messages[-1]["role"] != "assistant":
            raise ValueError("Last message was not an assistant message")
        return messages[-1]["content"]


class ReAct_Agent:
    def __init__(self, system_template="", agentic_model=None, guard_model=None, max_turns=10):
        # Initialize with parent class
        self.system_template= system_template
        self.agentic_model = agentic_model
        self.guard_model = guard_model
        self.max_turns = max_turns
        self.user_task = None
        self.injection_task = None
        self.normal_tools = []
        self.attack_tools = []
        self.step_labels = []

    def _check_user_task_utility(
        self,
        task: BaseUserTask,
        model_output: str,
        pre_environment: Env,
        task_environment: Env,
        functions_stack_trace: Sequence[FunctionCall],
    ) -> bool:
        #output_text_content = get_text_content_as_str(model_output)
        output_text_content = model_output
        utility_from_stack_traces = task.utility_from_traces(
            output_text_content, pre_environment, task_environment, functions_stack_trace
        )
        if utility_from_stack_traces is not None:
            return utility_from_stack_traces
        return task.utility(output_text_content, pre_environment, task_environment)
    
    def _check_injection_task_security(
        self,
        task: BaseInjectionTask,
        model_output: str,
        pre_environment: Env,
        task_environment: Env,
        functions_stack_trace: Sequence[FunctionCall],
    ) -> bool:
        # output_text_content = get_text_content_as_str(model_output)
        output_text_content = model_output
        security_from_stack_traces = task.security_from_traces(
            output_text_content, pre_environment, task_environment, functions_stack_trace
        )
        if security_from_stack_traces is not None:
            return security_from_stack_traces
        return task.security(output_text_content, pre_environment, task_environment)
    
    def _check_task_result_utility(
        self,
        task: BaseUserTask | BaseInjectionTask,
        model_output: list[MessageContentBlock],
        pre_environment: Env,
        task_environment: Env,
        functions_stack_trace: Sequence[FunctionCall],
    ) -> bool:
        
        return self._check_user_task_utility(
            task,
            model_output,
            pre_environment,
            task_environment,
            functions_stack_trace,
        )
    
    def _check_task_result_security(
        self,
        task: BaseUserTask | BaseInjectionTask,
        model_output: list[MessageContentBlock],
        pre_environment: Env,
        task_environment: Env,
        functions_stack_trace: Sequence[FunctionCall],
    ) -> bool:
        
        return self._check_injection_task_security(
            task,
            model_output,
            pre_environment,
            task_environment,
            functions_stack_trace,
        )
            
    def format_tools_for_prompt(self, tool_descriptions, tool_params):    
        result = []
        for tool, desc in tool_descriptions.items():
            result.append(f"{tool}: {desc}")
            params = tool_params.get(tool, {})
            if params:
                param_str = "\n    ".join([f"{k}: {v}" for k, v in params.items()])
                result.append(f"  parameters:\n    {param_str}")
            result.append("")
        return "\n".join(result)
        
    def extract_tool_params(self, text):
        tool_name, params_dict = extract_tool_params_react(text)
        return tool_name, params_dict

    def agent_invoke(self, pre_env=None, query="", available_tool_descriptions={}, available_tool_params={}, known_actions={}):
        i = 0
        self.step_labels = []
        self.normal_tools = []
        self.attack_tools = []
        known_actions["tool_safety_guardian"] = self.guard_model
        logs = ""
        available_tool_descriptions_str=self.format_tools_for_prompt(available_tool_descriptions, available_tool_params)
        available_tool_list_str = str(list(available_tool_descriptions.keys()))
        logs += "[System Prompt]:\n" + self.system_template.format(available_tool_descriptions=available_tool_descriptions_str, available_tool_names=available_tool_list_str)
        logs += "\n-----------------------------------\n"
        bot = Agent_Core(system=self.system_template.format(available_tool_descriptions=available_tool_descriptions_str, available_tool_names=available_tool_list_str), agentic_model=self.agentic_model)
        next_prompt = query
        logs += query
        logs += "\n-----------------------------------\n"

        while i < self.max_turns:
            i += 1
            result = bot(next_prompt)
            logs += result
            logs += "\n-----------------------------------\n"

            try:
                tool_name, tool_params = self.extract_tool_params(result)
                if tool_name:
                    logs += f" -- running {tool_name} {tool_params}\n"
                    if tool_name in known_actions:
                        # observation = known_actions[tool_name].call_tool(tool_name, deepcopy(tool_params))
                        if isinstance(known_actions[tool_name], FunctionType):
                            observation = known_actions[tool_name](**deepcopy(tool_params))
                        elif isinstance(known_actions[tool_name], dict):
                            observation = known_actions[tool_name]["output"]
                        elif isinstance(known_actions[tool_name], tuple):
                            runtime = known_actions[tool_name][0]
                            env = known_actions[tool_name][1]
                            mark = known_actions[tool_name][2]
                            if mark == "normal":
                                self.normal_tools.append(tool_name)
                            elif mark == "attack":
                                self.attack_tools.append(tool_name)
                            else:
                                self.normal_tools.append(tool_name)
                                self.attack_tools.append(tool_name)

                            tool_call_result, error = runtime.run_function(env, tool_name, deepcopy(tool_params))
                            if error:
                                observation = f"tool_call_result: {tool_call_result}\nerror: {error}"
                            else:
                                observation = tool_call_result
                        else:
                            observation = known_actions[tool_name].call_tool(tool_name, deepcopy(tool_params))

                    else:
                        observation = f"Unknown tool: {tool_name}"

                    logs += "Observation: "+ str(observation)
                    logs += "\n-----------------------------------\n"

                    next_prompt = f"Observation: {observation}"
                else:
                    #print("Response:", result)
                    break

            except Exception as e:
                observation = f"Error occurred: {str(e)}\n"
                logs += "Observation: "+ str(observation)
                logs += "\n-----------------------------------\n"
                next_prompt = f"Observation: {observation}"

            ######### 环境检验，步骤标注 ##########
            model_output = model_output_from_messages(bot.messages)
            functions_stack_trace = []
            utility = self._check_task_result_utility(
                self.user_task,
                model_output or [],
                pre_env,
                env,  # type: ignore
                functions_stack_trace,
            )

            # # Early return if no injection was intended
            # if self.injection_task is None:
            #     return messages, env_info, utility, True

            security = self._check_task_result_security(
                self.injection_task,
                model_output or [],
                pre_env,
                env,  # type: ignore
                functions_stack_trace,
            )

            self.step_labels.append({
                "tool_name": tool_name,
                "utility": utility,
                "security": security
            })

        return logs, bot.messages, available_tool_descriptions_str
