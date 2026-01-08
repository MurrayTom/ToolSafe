import re
import json,csv
import yaml
import pandas as pd
import sys
from tqdm import tqdm
from copy import deepcopy
import os
import argparse
import shutil
from openai import OpenAI

from task_executor.agentharm import *
from task_executor.asb import *
from task_executor.agentdojo_exec import *

from model.model import *
from agent.agent_prompts import *
from agent.default_agent import *
from agent.react_agent import *
from agent.react_firewall_agent import *
from agent.planexecute_agent import *
#from agent.ipiguard_agent import *
#from agent.react_agent_step_labeling import *
from agent.sec_react_agent import *
from agent.sec_planexecute_agent import *

from eval.llm_judge import *
from eval.rubrics_judge import *
from eval.rule_judge import *
from eval.bfcl_judge import *

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def build_evaluator(cfg, output_save_dir):
    """
    根据配置文件初始化评估器
    """
    judge_mode = cfg['judge']['mode']

    # 公共参数
    common_kwargs = {
        'judge_model_name': cfg['judge'].get('model_name', ''),
        'judge_model_path': cfg['judge'].get('model_path', ''),
        'judge_model_type': cfg['judge'].get('model_type', 'api'),
        'judge_api_base': cfg['judge'].get('api_base', ''),
        'judge_api_key': cfg['judge'].get('api_key', ''),
        'output_save_dir': output_save_dir
    }

    # 分支映射
    evaluator_map = {
        "llm_judge": lambda: LLM_Judge(**common_kwargs),
        "rubrics_judge": lambda: Rubrics_Judge(
            task_name=cfg['task']['name'],
            subset=cfg['task']['subset'],
            **common_kwargs
        ),
        "rule_based_judge": lambda: Rule_Judge(
            task_name=cfg['task']['name'],
            output_save_dir=output_save_dir
        ),
        "bfcl_judge": lambda: BFCL_Judge(
            task_name=cfg['task']['name'],
            output_save_dir=output_save_dir
        )
    }

    return evaluator_map.get(judge_mode, lambda: None)() if judge_mode in evaluator_map else None

def build_agent(cfg, agentic_model, guard_model):
    """
    根据配置文件初始化 Agent
    """
    agent_type = cfg['agent']['type']
    system_prompt_template = AGENT_PROMPT_TEMPLATES[cfg['agent'].get('system_prompt_template', '')]
    max_turns = cfg['agent'].get('max_turns', 10)

    # 分支映射
    agent_map = {
        'default': lambda: Default_Agent(
            system_template=system_prompt_template,
            agentic_model=agentic_model,
            guard_model=guard_model,
            max_turns=max_turns
        ),
        "react": lambda: ReAct_Agent(
            system_template=system_prompt_template,
            agentic_model=agentic_model,
            guard_model=guard_model,
            max_turns=max_turns
        ),
        "react_firewall": lambda: ReAct_Firewall_Agent(
            system_template=system_prompt_template,
            agentic_model=agentic_model,
            guard_model=guard_model,
            max_turns=max_turns
        ),
        "sec_react": lambda: SecReAct_Agent(
            system_template=system_prompt_template,
            agentic_model=agentic_model,
            guard_model=guard_model,
            max_turns=max_turns
        ),
        "plan_and_execute": lambda: PlanExecute_Agent(
            system_template=system_prompt_template,
            agentic_model=agentic_model,
            guard_model=guard_model,
            max_turns=max_turns
        ),
        "sec_plan_and_execute": lambda: SecPlanExecute_Agent(
            system_template=system_prompt_template,
            agentic_model=agentic_model,
            guard_model=guard_model,
            max_turns=max_turns
        )
    }

    agent = agent_map.get(agent_type)() if agent_type in agent_map else None
    
    return agent


if __name__ == "__main__":
    #***************************** argument setting *********************************************

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Agentic Security Benchmark Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    # 读取配置
    cfg = load_config(args.config)

    # 构建输出目录
    if "attack_type" in cfg['task']:
        output_save_dir = os.path.join(
            cfg['task']['output_dir'], 
            f"{cfg['task']['name']}_{cfg['task']['attack_type']}", 
            f"{cfg['model']['name']}_{cfg['agent']['type']}_{cfg['agent']['system_prompt_template']}"
        )
    elif "subset" in cfg['task']:
        output_save_dir = os.path.join(
            cfg['task']['output_dir'], 
            f"{cfg['task']['name']}_{cfg['task']['subset']}", 
            f"{cfg['model']['name']}_{cfg['agent']['type']}_{cfg['agent']['system_prompt_template']}"
        )
    else:
        output_save_dir = os.path.join(
            cfg['task']['output_dir'], 
            cfg['task']['name'], 
            f"{cfg['model']['name']}_{cfg['agent']['type']}_{cfg['agent']['system_prompt_template']}"
        )

    if cfg['guard'].get('enabled', False):
        output_save_dir += f"_{cfg['guard']['model_name']}"

    os.makedirs(output_save_dir, exist_ok=True)

    #***************************** argument setting *********************************************


    ############### Load Models ####################

    agentic_model = Model(
        model_name=cfg['model']['name'],
        model_path=cfg['model'].get('path', ''),
        model_type=cfg['model']['type'],
        api_base=cfg['model'].get('base_url', ''),
        api_key=cfg['model'].get('api_key', '')
    )

    if cfg['guard'].get('enabled', False):
        guard_model = Guardian(
            model_name=cfg['guard']['model_name'],
            api_base=cfg['guard'].get('base_url', ''),
            api_key=cfg['guard'].get('api_key', ''),
            model_path=cfg['guard'].get('path', ''), 
            model_type=cfg['guard'].get('type', '')
        )
    else:
        guard_model = None
    
    print("--------------------------------")


    ############### Build evaluator ####################

    evaluator = build_evaluator(cfg, output_save_dir)
    if evaluator is None:
        print(f"[Warning] Unknown judge mode: {cfg['judge']['mode']}")
    else:
        print(f"Evaluator {cfg['judge']['mode']} initialized successfully.")

    ############### Agent Settup ####################

    agent = build_agent(cfg, agentic_model, guard_model)
    if agent is None:
        raise ValueError(f"[Error] Unknown agent_type '{cfg['agent']['type']}' in config.")
    print(f"Agent '{cfg['agent']['type']}' initialized successfully.")

    ############### Task Start ####################
    
    if cfg['task'].get('inference_mode', True):
        if cfg['task'].get('name', "") == "agent-safetybench":
            task_processor = AgentSafetyBenchProcessor(
                agent=agent,
                output_save_dir=output_save_dir
            )
            meta_data = task_processor.run()
        
        if cfg['task'].get('name', "") == "agentharm":
            task_processor = AgentHarmProcessor(
                agent=agent,
                output_save_dir=output_save_dir,
                subtask=cfg['task'].get('subset', 'harmful')
            )
            meta_data = task_processor.run()

        if cfg['task'].get('name', "") == "asb":
            task_processor = ASBProcessor(
                agent=agent,
                output_save_dir=output_save_dir,
                attack_type=cfg['task'].get('attack_type', "OPI")
            )
            meta_data = task_processor.run(cfg['task'].get('task_nums', 5))

        if cfg['task'].get('name', "") == "agentdojo":
            task_processor = AgentdojoProcessor(
                agent=agent,
                output_save_dir=output_save_dir,
                attack_type=cfg['task'].get('attack_type', ""),
                environments = cfg['task']["environment"]
            )
            meta_data = task_processor.run(cfg['task']["user_task"])
        

    else:
        meta_data = []
        meta_data_path = os.path.join(output_save_dir, "meta_data.json")
        if os.path.exists(meta_data_path):
            with open(meta_data_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)

    
    ############### Evaluation ####################
    
    if evaluator:
        for i in range(len(meta_data)):
            meta_data[i]["agent_type"] = cfg['agent']['type']

        eval_results, eval_results_meta = evaluator.judge(meta_data)

        total = len(eval_results)
        safe_count = eval_results.count("safe")
        unsafe_count = eval_results.count("unsafe")
        format_error_count = eval_results.count("format error")

        safe_ratio = safe_count / total
        unsafe_ratio = unsafe_count / total
        format_error_ratio = format_error_count / total

        print(f"Safe: {safe_count} ({safe_ratio:.2%})")
        print(f"Unsafe: {unsafe_count} ({unsafe_ratio:.2%})")
        print(f"Format Error: {format_error_count} ({format_error_ratio:.2%})")

        stats = {
            "total": total,
            "safe": {
                "count": safe_count,
                "ratio": safe_ratio
            },
            "unsafe": {
                "count": unsafe_count,
                "ratio": unsafe_ratio
            },
            "format_error": {
                "count": format_error_count,
                "ratio": format_error_ratio
            }
        }

        with open(os.path.join(output_save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        with open(os.path.join(output_save_dir, "eval_results_meta.json"), "w", encoding="utf-8") as f:
            json.dump(eval_results_meta, f, ensure_ascii=False, indent=4)

