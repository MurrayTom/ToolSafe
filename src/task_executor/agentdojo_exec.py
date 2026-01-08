import os
import importlib.util
import json
import csv
import sys
sys.path.append('./benchmark/agentharm')
from openai import OpenAI
from agent.react_agent import *
from agent.sec_react_agent import *
from model.model import *

import importlib
import logging
import warnings
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import click 
from dotenv import load_dotenv
from rich import print
from rich.live import Live
from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite, get_suites
from agentdojo.task_suite.task_suite import TaskSuite

import inspect

class AgentdojoProcessor:
    def __init__(self, agent, output_save_dir, attack_type, environments):
        self.agent = agent
        self.attack_type = attack_type
        self.suites = environments
        self.benchmark_version = "v1.2.2"
        self.output_save_dir = output_save_dir

        self.agent_pipeline = AgentPipeline.from_config(
            PipelineConfig(
                agent=self.agent,
                model_id = None,
                defense="tool_filter", #"tool_filter"
                system_message_name=None,
                system_message=None,
                tool_output_format=None,
            )
        )

    def benchmark_suite(
        self,
        suite: TaskSuite,
        logdir: str,
        user_tasks: tuple[str, ...] = ()
    ) -> SuiteResults:
        
        if self.attack_type is None:
            results = benchmark_suite_without_injections(
                self.agent_pipeline,
                suite,
                user_tasks=user_tasks,
                logdir=logdir,
                force_rerun=False,
                benchmark_version=self.benchmark_version,
            )
        else:
            attacker_ = load_attack(self.attack_type, suite, self.agent_pipeline)
            meta_data, results = benchmark_suite_with_injections(
                self.agent_pipeline,
                suite,
                attacker_,
                user_tasks=user_tasks,
                injection_tasks=None,
                logdir=logdir,
                force_rerun=False,
                benchmark_version=self.benchmark_version,
            )
        print(f"Finished benchmark for suite: '{suite.name}'")

        return meta_data, results
    
    def show_results(self, suite_name: str, results: SuiteResults, show_security_results: bool):
        utility_results = results["utility_results"].values()
        avg_utility = sum(utility_results) / len(utility_results)
        avg_security = None

        print(f"Results for suite {suite_name}")
        print(f"Average utility: {avg_utility * 100:.2f}%")

        if show_security_results:
            passed_injection_tasks = sum(results["injection_tasks_utility_results"].values())
            total_injection_tasks = len(results["injection_tasks_utility_results"])
            print(f"\nPassed injection tasks as user tasks: {passed_injection_tasks}/{total_injection_tasks}")

            security_results = results["security_results"].values()
            avg_security = sum(security_results) / len(security_results)
            print(f"Average security: {avg_security * 100:.2f}%")

        return avg_utility, avg_security

    
    def run(self, user_tasks):
        results = {}
        
        for suite_name in self.suites:

            output_save_path = f"{self.output_save_dir}/{suite_name}"
            os.makedirs(output_save_path, exist_ok=True)

            ############## 环境
            suite = get_suite(self.benchmark_version, suite_name) ##############
            meta_data, results[suite_name] = self.benchmark_suite(suite, output_save_path, user_tasks)

            with open(output_save_path+"/meta_data.json", "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)

        metrics = {}   # 用于存全部 suite 的指标
        combined_results = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})
        for suite_name, result in results.items():
            avg_utility, avg_security = self.show_results(suite_name, result, self.attack_type is not None)
            metrics[suite_name] = {
                "avg_utility": avg_utility,
                "avg_security": avg_security
            }
            for user_task, injection_task in result["utility_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["utility_results"][(new_user_task, injection_task)] = result["utility_results"][
                    (user_task, injection_task)
                ]
            for user_task, injection_task in result["security_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["security_results"][(new_user_task, injection_task)] = result["security_results"][
                    (user_task, injection_task)
                ]
            for injection_task in result["injection_tasks_utility_results"]:
                new_injection_task = suite_name + "_" + injection_task
                combined_results["injection_tasks_utility_results"][new_injection_task] = result[
                    "injection_tasks_utility_results"
                ][injection_task]
        avg_utility, avg_security = self.show_results("combined", combined_results, self.attack_type is not None)
        metrics["combined"] = {
                "avg_utility": avg_utility,
                "avg_security": avg_security
            }
        with open(os.path.join(self.output_save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

