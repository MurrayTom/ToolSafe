import os
import re
import json,csv
import yaml
import pandas as pd
import sys
from tqdm import tqdm
from copy import deepcopy
import argparse
import shutil
from openai import OpenAI

from guardian_evaluator.agentharm import *
from guardian_evaluator.asb import *
from guardian_evaluator.agentdojo import *

from model.model import *
from agent.agent_prompts import *

from utils.guardian_score_mapping import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

TASK_PROCESSOR_MAP = {
    "agentharm": AgentHarmProcessor,
    "asb": ASBProcessor,
    "agentdojo": AgentDojoProcessor,
}

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def build_guard_model(cfg):
    model_cfg = cfg["model"]
    if model_cfg["type"] == "local":
        return Guardian(
            model_name=model_cfg["name"],
            model_path=model_cfg["path"],
            model_type="local"
        )
    else:
        return Guardian(
            model_name=model_cfg["name"],
            api_base=model_cfg["api"]["base_url"],
            api_key=model_cfg["api"]["api_key"],
            model_type="api"
        )
    
def build_output_dir(cfg):
    root = cfg["experiment"]["output_root"]
    task = cfg["task"]["name"]
    model = cfg["model"]["name"]
    attack = cfg["task"].get("attack_type", "")

    if attack:
        path = os.path.join(root, task, attack, model)
    else:
        path = os.path.join(root, task, model)

    os.makedirs(path, exist_ok=True)
    return path
    

def load_cached_results(output_save_dir):
    meta_data, preds, labels = [], [], []
    meta_data_path = os.path.join(output_save_dir, "meta_data.json")
    preds_path = os.path.join(output_save_dir, "preds.json")
    labels_path = os.path.join(output_save_dir, "labels.json")

    if os.path.exists(meta_data_path):
        with open(meta_data_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
    
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as f:
            preds = json.load(f)

    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    return meta_data, preds, labels


def run_task(cfg, guard_model, output_dir):
    task_cfg = cfg["task"]
    task_name = task_cfg["name"]
    attack_type = task_cfg.get("attack_type", "")

    processor_cls = TASK_PROCESSOR_MAP[task_name]
    processor = processor_cls(
        guard_model=guard_model,
        output_save_dir=output_dir
    )

    if task_name in ["agentharm", "agentdojo"]:
        meta_data, preds, labels = processor.run()
    elif task_name in ["asb"]:
        meta_data, preds, labels = processor.run(attack_type)
    else:
        print(f"The task {task_name} is not supported now ...")
        exit()

    return meta_data, preds, labels


def compute_metrics(preds_clean, labels_clean, score_mode):
    if len(labels_clean) > 0:
        acc = accuracy_score(labels_clean, preds_clean)
        if score_mode == "exact":
            f1 = f1_score(labels_clean, preds_clean, average="macro")
            recall = recall_score(labels_clean, preds_clean, average="macro")
        else:
            f1 = f1_score(labels_clean, preds_clean, average="binary")
            recall = recall_score(labels_clean, preds_clean, average="binary")
    else:
        acc, f1, recall = 0.0, 0.0, 0.0

    print("Accuracy:", acc)
    print("F1:", f1)
    print("Recall:", recall)

    stats = {
        "total": len(labels_clean),
        "accuracy": acc,
        "f1": f1,
        "recall": recall
    }

    return stats

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = build_output_dir(cfg)
    guard_model = build_guard_model(cfg)

    if cfg["experiment"]["inference_mode"]:
        meta_data, preds, labels = run_task(cfg, guard_model, output_dir)
    else:
        meta_data, preds, labels = load_cached_results(output_dir)

    preds_clean, labels_clean = filter_valid_pairs(
        preds, labels,
        score_mode=cfg["experiment"]["score_mode"]
    )

    metrics = compute_metrics(
        preds_clean, labels_clean,
        score_mode=cfg["experiment"]["score_mode"]
    )

    with open(os.path.join(output_dir, f"metrics_{cfg['experiment']['score_mode']}.json"), "w") as f:
        json.dump(metrics, f, indent=4)