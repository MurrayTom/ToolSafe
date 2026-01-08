import json
import csv
import sys
import os
import time

from agent.agent_prompts import *

class AgentHarmProcessor:
    def __init__(self, guard_model, output_save_dir):
        self.guard_model = guard_model
        self.data_path = ["./TS-Bench/agentharm-traj/harmful_steps.json", "./TS-Bench/agentharm-traj/benign_steps.json"]
        self.output_save_dir = output_save_dir
        self.guard_prompt_template = GUARD_TEMPLATES[self.guard_model.model_name]
        
    def run(self):
        data = []
        for i in range(len(self.data_path)):
            with open(self.data_path[i], 'r') as f:
                data.extend(json.load(f))
            print(f"the number of samples in agentharm (trajectory) dataset:", len(data))

        meta_data, preds, labels = [], [], []

        output_save_path = f"{self.output_save_dir}/meta_data.json"
        output_save_pred_path = f"{self.output_save_dir}/preds.json"
        output_save_label_path = f"{self.output_save_dir}/labels.json"
        
        if os.path.exists(output_save_path):
            with open(output_save_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

        if os.path.exists(output_save_pred_path):
            with open(output_save_pred_path, "r", encoding="utf-8") as f:
                preds = json.load(f)

        if os.path.exists(output_save_label_path):
            with open(output_save_label_path, "r", encoding="utf-8") as f:
                labels = json.load(f)

        assert len(meta_data) == len(preds) == len(labels)

        for _, sample in enumerate(data[len(meta_data):]):
            user_query = sample['instruction']
            agent_logs = {
                "interaction_history": sample['history'],
                "current_action": sample['current_action']
            }
            env_info = sample['env_info']

            judge_res = self.guard_model.get_judgment_res(meta_info={
                "env_info" : env_info,
                "user_request": user_query,
                "agent_action": agent_logs
            })
            
            result = {
                "meta_sample": sample,
                "guard_res": judge_res
            }
            meta_data.append(result)

            if "risk rating" in judge_res:
                preds.append(judge_res["risk rating"])
            else:
                preds.append(None)
            labels.append(sample['score'])

            # Save intermediate results
            with open(output_save_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)

            with open(output_save_pred_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, ensure_ascii=False, indent=4)
            
            with open(output_save_label_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=4)
        
        return meta_data, preds, labels
    