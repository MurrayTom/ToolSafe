import os
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
from vllm.sampling_params import SamplingParams
from agent.agent_prompts import *
from utils.guardian_parser import *

import httpx
user = "xxx"
pwd  = "xxx"
auth = httpx.BasicAuth(user, pwd)

class Model:
    def __init__(self, model_name="gpt-4o", model_path="", model_type="api", api_base="", api_key =""):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path

        if self.model_type == "api":
            if "gpt" in self.model_name.lower() or "claude" in self.model_name.lower() or "gemini" in self.model_name.lower():
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=api_base
                )
            else:
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=httpx.Client(auth=auth, verify=True)
                )
        elif self.model_type == "analysis":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )

            self.llm.eval()

        else:
            self.llm = LLM(model=model_path)
            self.sampling = SamplingParams(max_tokens=2048, temperature=0.1, top_p=0.9)



class Guardian:
    def __init__(self, model_name="gpt-4o", model_path="", model_type="api", api_base="", api_key =""):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path

        if self.model_type == "api":
            if "gpt" in self.model_name.lower() or "claude" in self.model_name.lower() or "gemini" in self.model_name.lower():
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=api_base
                )
            else:
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=httpx.Client(auth=auth, verify=True)
                )
        else:
            self.llm = LLM(model=model_path)
            self.sampling = SamplingParams(max_tokens=2048, temperature=0.1, top_p=0.9)


    def get_judgment_res(self, meta_info, max_turn=3):
        guard_input = GUARD_TEMPLATES[self.model_name].format(**meta_info)
        guard_messages = [{"role": "user", "content": guard_input}]
        results = {}
        j = 0
        while(j<max_turn):
            j+=1
            if self.model_type == "api":
                print("=================================")
                print(self.model_name)
                guard_res = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=guard_messages
                )
                print(guard_res)
                guard_res = guard_res.choices[0].message.content
            else:
                guard_res = self.llm.chat(guard_messages, sampling_params=self.sampling)
                guard_res = guard_res[0].outputs[0].text.strip()

            if self.model_name == "ashell-guardian" or self.model_name == "TS-Guard" or self.model_name == "qwen2.5-7b-instruct" or self.model_name == "qwen3-8b" or "gpt" in self.model_name:
                parser_res, results = guardian_paser_map[self.model_name](guard_res)
            elif self.model_name == "ashell-guardian-sftonly": #or self.model_name == "qwen2.5-7b-instruct"
                parser_res, results = guardian_paser_map[self.model_name](guard_res)
            else:
                parser_res = guardian_paser_map[self.model_name](guard_res)

            if parser_res in [0,1,0.5]:
                return {"risk rating": parser_res, "results": results, "reason": guard_res}
            else:
                continue
        
        return {"reason": guard_res}
    
    def call_tool(self, tool_name, arguments):
        if tool_name == "tool_safety_guardian":
            guardian_res = self.tool_safety_guardian(**arguments)
            return guardian_res

        elif tool_name == "alignment_check":
            alignment_res = self.alignment_check(**arguments)
            return alignment_res
        
        else:
            return "tool name error, the correct name is tool_safety_guardian"

    
    def tool_safety_guardian(self, 
                     user_request, 
                     interaction_history, 
                     current_action, 
                     current_action_description):
        guard_input = GUARD_TEMPLATES[self.model_name].format(env_info=current_action_description,
                                                              user_request=user_request,
                                                              agent_action={"interaction_history": interaction_history,
                                                                            "current_action": current_action})

        guard_messages = [{"role": "user", "content": guard_input}]
        results = {}

        j = 0
        while(j<3):
            j+=1
            if self.model_type == "api":
                guard_res = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=guard_messages
                )
                guard_res = guard_res.choices[0].message.content
            else:
                guard_res = self.llm.chat(guard_messages, sampling_params=self.sampling)
                guard_res = guard_res[0].outputs[0].text.strip()

            if self.model_name == "ashell-guardian" or self.model_name == "TS-Guard": #or self.model_name == "qwen2.5-7b-instruct"
                parser_res, results = guardian_paser_map[self.model_name](guard_res)
            else:
                parser_res = guardian_paser_map[self.model_name](guard_res)

            if parser_res in [0,1,0.5]:
                return {"risk rating": parser_res, "results": results, "reason": guard_res}
            else:
                continue

        return {"reason": guard_res}
    

    def alignment_check(self, 
                     user_request, 
                     interaction_history, 
                     current_action, 
                     current_action_description):
        system_prompt, user_prompt = GUARD_TEMPLATES["alignmentcheck"][0], GUARD_TEMPLATES["alignmentcheck"][1].format(env_info=current_action_description,
                                                              user_request=user_request,
                                                              agent_action={"interaction_history": interaction_history,
                                                                            "current_action": current_action})

        guard_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        j = 0
        while(j<3):
            j+=1
            try:
                if self.model_type == "api":
                    guard_res = self.llm.chat.completions.create(
                        model=self.model_name,
                        messages=guard_messages,
                        response_format={"type": "json_object"}
                    )
                    guard_res = guard_res.choices[0].message.content
                else:
                    guard_res = self.llm.chat(guard_messages, sampling_params=self.sampling)
                    guard_res = guard_res[0].outputs[0].text.strip()

                print("=================================")
                print(guard_res)
                print("=================================")
                parser_res = alignment_check_parser(guard_res)

                return {"alignment_check_passed": not parser_res, "reason": guard_res}
            except Exception as e:
                print(f"Alignment check error: {str(e)}")
                continue

        return {"alignment_check_passed": not parser_res, "reason": guard_res}