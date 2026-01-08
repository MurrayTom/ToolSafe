import os
import json
from openai import OpenAI
from vllm import LLM
from vllm.sampling_params import SamplingParams
from agent.agent_prompts import *
from utils.guardian_parser import *
from utils.guardian_parser import *

import httpx
user = "88014349af321d3f0d552d4919f749d5"
pwd  = "7ae160b38f6ee9ac359766c722b854a5"
auth = httpx.BasicAuth(user, pwd)

class LLM_Judge:
    def __init__(self, judge_model_name="shieldagent", judge_model_path="", judge_model_type="api", judge_api_base="", judge_api_key ="", output_save_dir=""):
        self.judge_model_name = judge_model_name
        self.judge_template = GUARD_TEMPLATES[self.judge_model_name]
        self.output_save_dir = output_save_dir
        if judge_model_type == "api":
            if "gpt" in self.judge_model_name.lower() or "claude" in self.judge_model_name.lower():
                self.judge_model = OpenAI(
                    api_key=judge_api_key,
                    base_url=judge_api_base
                )
            else:
                self.judge_model = OpenAI(
                    api_key=judge_api_key,
                    base_url=judge_api_base,
                    http_client=httpx.Client(auth=auth, verify=True)
                )
        else:
            self.judge_model = LLM(model=judge_model_path)
            self.sampling = SamplingParams(max_tokens=2048, temperature=0.1, top_p=0.9)

    def judge(self, meta_data):
        print("The number of agent logs:", len(meta_data))
        eval_results = []
        eval_results_meta = []

        output_eval_results_path = f"{self.output_save_dir}/eval_results.json"
        output_eval_results_meta_path = f"{self.output_save_dir}/eval_results_meta.json"

        if os.path.exists(output_eval_results_path):
            with open(output_eval_results_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

        if os.path.exists(output_eval_results_meta_path):
            with open(output_eval_results_meta_path, "r", encoding="utf-8") as f:
                eval_results_meta = json.load(f)

        assert len(eval_results) == len(eval_results_meta)

        for item in meta_data[len(eval_results):]:
            judge_meta_info = {"user_request": item["messages"][1]["content"], "agent_action": item["messages"][2:], "env_info": item["env_info"]}
            judge_input = self.judge_template.format(**judge_meta_info)

            judge_res = self.judge_model.chat.completions.create(
                model=self.judge_model_name,
                messages=[{"role": "user", "content": judge_input}]
            )
            judge_res = judge_res.choices[0].message.content

            parser_res = llm_judge_parser(judge_res)

            eval_results.append(parser_res)
            eval_results_meta.append({"meta_data": item, "judge_output": judge_res, "eval": parser_res})

            with open(output_eval_results_path, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=4)
            with open(output_eval_results_meta_path, "w", encoding="utf-8") as f:
                json.dump(eval_results_meta, f, ensure_ascii=False, indent=4)

        return eval_results, eval_results_meta

            