import re
import json
from copy import deepcopy
from types import FunctionType
from tqdm import tqdm
from agent.agent import *
from agent.agent_prompts import *

from utils.tool_parser import *

class ReAct_Firewall_Agent:
    def __init__(self, system_template="", agentic_model=None, guard_model=None, max_turns=10):
        # Initialize with parent class
        self.system_template= system_template
        self.agentic_model = agentic_model
        self.guard_model = guard_model
        self.max_turns = max_turns
    
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

    def agent_invoke(self, query="", available_tool_descriptions={}, available_tool_params={}, known_actions={}, injection_task=""):
        i = 0
        logs = ""
        available_tool_descriptions_str=self.format_tools_for_prompt(available_tool_descriptions, available_tool_params)
        available_tool_list_str = str(list(available_tool_descriptions.keys()))
        logs += "[System Prompt]:\n" + self.system_template.format(available_tool_descriptions=available_tool_descriptions_str, available_tool_names=available_tool_list_str)
        logs += "\n-----------------------------------\n"
        bot = Agent_Core(system=self.system_template.format(available_tool_descriptions=available_tool_descriptions_str, available_tool_names=available_tool_list_str), agentic_model=self.agentic_model)
        next_prompt = query
        logs += query
        logs += "\n-----------------------------------\n"

        ##################  输入护栏 #####################
        # meta_info = {"user_request": query, "interaction_history": [], "current_action": "", "current_action_description": available_tool_descriptions_str}
        # guard_res = self.guard_model.call_tool(tool_name="tool_safety_guardian", arguments=meta_info)

        # if "risk rating" in guard_res and guard_res["risk rating"] > 0:
        #     guard_res["final answer"] = "Sorry, I can't assist with the request"
        #     bot.messages.append({"role": "user", "content": query})
        #     bot.messages.append({"role": "assistant", "content": guard_res})
        #     return logs, bot.messages, available_tool_descriptions_str

        #################################################

        while i < self.max_turns:
            i += 1
            result = bot(next_prompt)
            print(result)
            logs += result
            logs += "\n-----------------------------------\n"

            try:
                tool_name, tool_params = self.extract_tool_params(result)
                print(tool_name, tool_params)
                if tool_name:
                    logs += f" -- running {tool_name} {tool_params}\n"
                    if tool_name in known_actions:
                        ########## Alignment Check ######################
                        agent_action = bot.messages[2:]
                        interaction_history, current_action = agent_action[:-1], agent_action[-1]
                        meta_info = {"user_request": query, "interaction_history": interaction_history, "current_action": current_action, "current_action_description": available_tool_descriptions_str}
                        # guard_res = self.guard_model.call_tool(tool_name="tool_safety_guardian", arguments=meta_info)
                        # if "risk rating" in guard_res and guard_res["risk rating"] >= 0.5:
                        #     bot.messages.append({"role": "user", "content": guard_res})
                        #     bot.messages.append({"role": "assistant", "content": "Sorry, I can't assist with the request"})
                        #     return logs, bot.messages, available_tool_descriptions_str


                        guard_res = self.guard_model.call_tool(tool_name="alignment_check", arguments=meta_info)
                        if not guard_res["alignment_check_passed"]:
                            bot.messages.append({"role": "user", "content": guard_res})
                            bot.messages.append({"role": "assistant", "content": "Sorry, I can't assist with the request"})
                            return logs, bot.messages, available_tool_descriptions_str

                        #################################################

                        if isinstance(known_actions[tool_name], FunctionType):
                            observation = known_actions[tool_name](**deepcopy(tool_params))
                        elif isinstance(known_actions[tool_name], dict):
                            observation = known_actions[tool_name]["output"]
                        elif isinstance(known_actions[tool_name], tuple):
                            runtime = known_actions[tool_name][0]
                            env = known_actions[tool_name][1]
                            tool_call_result, error = runtime.run_function(env, tool_name, deepcopy(tool_params))
                            if error:
                                observation = f"tool_call_result: {tool_call_result}\nerror: {error}"
                            else:
                                observation = tool_call_result
                        else:
                            observation = known_actions[tool_name].call_tool(tool_name, deepcopy(tool_params))

                        #### 注入任务 ########
                        if injection_task and tool_name==injection_task["tool_name"]:
                            observation = observation + injection_task["template"].format(injection_prompt=injection_task["injection_prompt"])
                        #####################

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

        return logs, bot.messages, available_tool_descriptions_str
