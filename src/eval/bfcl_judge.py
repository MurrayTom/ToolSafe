import os
import json
import sys
import numpy as np
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import statistics
from collections import defaultdict

from task_executor.bfcl.utils import (
    extract_test_category,
    load_file,
    load_dataset_entry,
    load_ground_truth_entry,
    write_list_of_dicts_to_file,
    get_directory_structure_by_category,
    parse_prompt_variation_params,
    is_empty_output,
    is_function_calling_format_output,
    is_empty_execute_response,
)
from task_executor.bfcl.constants import (
    VERSION_PREFIX,
    is_relevance_or_irrelevance,
    is_format_sensitivity,
    is_agentic,
    is_multi_turn,
    is_chatable,
    is_sql,
    is_executable,
    is_memory_prereq,
    is_java,
    is_js,
)
from task_executor.bfcl.parser import (
    ReturnFormat,
    Language,
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from task_executor.bfcl.eval_checker.ast_eval.ast_checker import (
    ast_checker,
)
from task_executor.bfcl.eval_checker.agentic_eval.agentic_checker import (
    agentic_checker,
)
from task_executor.bfcl.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

rule_supported_task = ["bfcl"]

RESULT_FILE_PATTERN = f"{VERSION_PREFIX}_*_result.json"

class BFCL_Judge:
    def __init__(self, task_name="bfcl", output_save_dir=""):
        self.output_save_dir = output_save_dir
        if task_name not in rule_supported_task:
            print(task_name + "is not supported by the evaluation mode")
            exit()
        # TODO: 参数化
        self.allow_missing = False
        self.is_fc_model = False

    def check_attack_success(self, messages, attacker_goal):
        attack_success = False

        for message in messages:
            content = message['content']
            if content is not None and attacker_goal in content:
                attack_success = True
                break

        return attack_success

    def check_original_success(self, messages, normal_tools):
        found_tools = set()

        for message in messages:
            content = message['content']
            if content:
                for instance in normal_tools:
                    if instance["Expected Achievements"] in content:
                        found_tools.add(instance["Tool Name"])

        success = all(tool["Tool Name"] in found_tools for tool in normal_tools)

        found_count = sum(1 for tool in normal_tools if tool["Tool Name"] in found_tools)
        score = found_count / len(normal_tools) if normal_tools else 0

        return success, score

    def judge(self, meta_data):
        print("The number of agent logs:", len(meta_data))

        result_dir = Path(self.output_save_dir)
        score_dir = result_dir / "eval_results_meta"
        score_dir.mkdir(parents=True, exist_ok=True)

        metrics = []

        output_eval_metrics_path = f"{self.output_save_dir}/metrics.json"

        for model_result_json in result_dir.rglob(RESULT_FILE_PATTERN):
            test_category = extract_test_category(model_result_json)

            # We don't evaluate the following categories in the current iteration of the benchmark
            if (
                is_chatable(test_category)
                or is_sql(test_category)
                or is_executable(test_category)
                or is_memory_prereq(test_category)
            ):
                continue

            model_result = load_file(model_result_json, sort_by_id=True)

            accuracy, eval_result = self.evaluate_task(
                test_category,
                result_dir,
                score_dir,
                model_result,
            )

            metrics.append({
                "test_category": test_category,
                "accuracy": accuracy,
                "total_count": len(eval_result),
            })

            self.save_eval_results(eval_result, test_category, score_dir)

        with open(output_eval_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        return [], []

    def evaluate_task(
        self,
        test_category,
        result_dir,
        score_dir,
        model_result,
    ):
        print(f"🔍 Running test: {test_category}")

        # Find the corresponding prompt entries
        prompt = load_dataset_entry(
            test_category, include_prereq=False, include_language_specific_hint=False
        )

        if is_relevance_or_irrelevance(test_category):
            prompt, _ = self._subset_entries_by_model_ids(
                model_result, prompt, None
            )

            accuracy, eval_result = self.relevance_file_runner(
                model_result, prompt, test_category, score_dir
            )

        else:
            # Find the corresponding possible answer entries
            possible_answer = load_ground_truth_entry(test_category)
            # Sanity: prompt and ground truth should be 1:1
            assert len(prompt) == len(
                possible_answer
            ), f"Length of ground truth ({len(possible_answer)}) should match prompt entries ({len(prompt)})."

            prompt, possible_answer = self._subset_entries_by_model_ids(
                model_result, prompt, possible_answer
            )

            if is_format_sensitivity(test_category):
                accuracy, eval_result = self.format_sensitivity_runner(
                    model_result,
                    prompt,
                    possible_answer,
                    test_category,
                    score_dir,
                )

            elif is_multi_turn(test_category):
                accuracy, eval_result = self.multi_turn_runner(
                    model_result,
                    prompt,
                    possible_answer,
                    test_category,
                    score_dir,
                )

            elif is_agentic(test_category):
                accuracy, eval_result = self.agentic_runner(
                    model_result,
                    prompt,
                    possible_answer,
                    test_category,
                    score_dir,
                )
            # Single turn test
            else:
                accuracy, eval_result = self.ast_file_runner(
                    model_result,
                    prompt,
                    possible_answer,
                    test_category,
                    score_dir,
                )

        print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy:.2%}")

        return accuracy, eval_result

    def relevance_file_runner(
        self, 
        model_result, 
        prompt, 
        test_category, 
        score_dir,
    ):
        # This function serves for both relevance and irrelevance tests, which share the exact opposite logic.
        # If `test_category` is "irrelevance", the model is expected to output no function call.
        # No function call means either the AST decoding fails (a error message is generated) or the decoded AST does not contain any function call (such as a empty list, `[]`).
        # If `test_category` is "relevance", the model is expected to output to a function call, and empty list doesn't count as a function call.
        result = []
        correct_count = 0
        for i in range(len(model_result)):
            index = model_result[i]["id"]
            model_result_item = model_result[i]["result"]
            prompt_entry = prompt[i]

            entry_result = self._evaluate_single_relevance_entry(
                index, model_result_item, prompt_entry, test_category
            )

            if entry_result["valid"]:
                correct_count += 1
            result.append(entry_result)

        return correct_count / len(model_result), result

    def format_sensitivity_runner(
            self,
            model_result,
            prompt,
            possible_answer,
            test_category,
            score_dir,
        ):
            assert (
                len(model_result) == len(prompt) == len(possible_answer)
            ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

            # The format sensitivity tests are all single-turn tests, so we use a similar logic to the ast_file_runner to evaluate them.

            result = []
            correct_count = 0
            # Track stats per format sensitivity configuration
            config_stats: dict[str, dict[str, int]] = defaultdict(
                lambda: {"correct": 0, "total": 0}
            )

            for i in range(len(model_result)):
                index = model_result[i]["id"]
                model_result_item = model_result[i]["result"]
                prompt_entry = prompt[i]
                possible_answer_item = possible_answer[i]["ground_truth"]

                assert (
                    ":" in index and len(index.split(":")) == 3
                ), f"Test entry ID {index} should contain exactly two colons, since they are supposed to be the format sensitivity ids."

                format_sensitivity_config = index.split(":")[1]
                (
                    return_format,
                    has_tool_call_tag,
                    function_doc_format,
                    prompt_format,
                    prompt_style,
                ) = parse_prompt_variation_params(format_sensitivity_config)

                return_format = ReturnFormat(return_format)

                entry_result = self._evaluate_single_ast_entry(
                    index,
                    model_result_item,
                    possible_answer_item,
                    prompt_entry,
                    test_category,
                    # Format sensitivity tests are all python tests
                    language=Language.PYTHON,
                    return_format=return_format,
                    has_tool_call_tag=has_tool_call_tag,
                )

                # Update stats for this configuration
                config_stats[format_sensitivity_config]["total"] += 1
                if entry_result["valid"]:
                    correct_count += 1
                    config_stats[format_sensitivity_config]["correct"] += 1
                result.append(entry_result)

            # Compute accuracy per configuration
            accuracy_by_config = {
                cfg: {
                    "accuracy": stats["correct"] / stats["total"],
                    "correct_count": stats["correct"],
                    "total_count": stats["total"],
                }
                for cfg, stats in config_stats.items()
            }

            # Calculate statistics across different prompt configurations
            config_accuracies = [v["accuracy"] for v in accuracy_by_config.values()]
            if len(config_accuracies) > 1:
                accuracy_variance = round(statistics.variance(config_accuracies) * 100**2, 2)
                accuracy_std = round(statistics.stdev(config_accuracies) * 100, 2)
                accuracy_max_delta = round(
                    (max(config_accuracies) - min(config_accuracies)) * 100, 2
                )
            else:
                accuracy_variance = 0.0
                accuracy_std = 0.0
                accuracy_max_delta = 0.0

            extra_header_fields = {
                "accuracy_max_delta": accuracy_max_delta,
                "accuracy_variance": accuracy_variance,
                "accuracy_std": accuracy_std,
                **accuracy_by_config,
            }

            return correct_count / len(model_result), result

    def multi_turn_runner(
        self,
        model_result,
        prompt,
        possible_answer,
        test_category,
        score_dir,
    ):
        assert (
            len(model_result) == len(prompt) == len(possible_answer)
        ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

        result = []
        correct_count = 0
        for i in range(len(model_result)):
            index = model_result[i]["id"]
            multi_turn_model_result_list = model_result[i]["result"]
            multi_turn_ground_truth_list = possible_answer[i]["ground_truth"]
            test_entry = prompt[i]

            entry_result = self._evaluate_single_multi_turn_entry(
                index,
                multi_turn_model_result_list,
                multi_turn_ground_truth_list,
                test_entry,
                test_category,
            )

            if entry_result["valid"]:
                correct_count += 1
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)

        return correct_count / len(model_result), result

    def agentic_runner(
        self,
        model_result,
        prompt,
        possible_answer,
        test_category,
        score_dir,
    ):
        assert (
            len(model_result) == len(prompt) == len(possible_answer)
        ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

        result = []
        correct_count = 0
        for i in range(len(model_result)):
            index = model_result[i]["id"]
            model_result_list = model_result[i]["result"]
            possible_answer_item = possible_answer[i]["ground_truth"]
            test_entry = prompt[i]

            entry_result = self._evaluate_single_agentic_entry(
                index,
                model_result_list,
                possible_answer_item,
                test_entry,
                test_category,
            )

            if entry_result["valid"]:
                correct_count += 1
            entry_result["inference_log"] = model_result[i].get("inference_log", "")
            result.append(entry_result)
                
        return correct_count / len(model_result), result

    def ast_file_runner(
        self,
        model_result,
        prompt,
        possible_answer,
        test_category,
        score_dir,
    ):
        assert (
            len(model_result) == len(prompt) == len(possible_answer)
        ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

        if is_java(test_category):
            language = Language.JAVA
            return_format = ReturnFormat.JAVA
        elif is_js(test_category):
            language = Language.JAVASCRIPT
            return_format = ReturnFormat.JAVASCRIPT
        else:
            language = Language.PYTHON
            return_format = ReturnFormat.PYTHON

        result = []
        correct_count = 0
        for i in range(len(model_result)):
            index = model_result[i]["id"]
            model_result_item = model_result[i]["result"]
            prompt_entry = prompt[i]
            possible_answer_item = possible_answer[i]["ground_truth"]

            entry_result = self._evaluate_single_ast_entry(
                index,
                model_result_item,
                possible_answer_item,
                prompt_entry,
                test_category,
                language=language,
                return_format=return_format,
                has_tool_call_tag=False,
            )

            if entry_result["valid"]:
                correct_count += 1
            result.append(entry_result)

        return correct_count / len(model_result), result

    def save_eval_results(
        self,
        result,
        test_category,
        score_dir,
    ) -> tuple[float, int]:
        """
        Compute accuracy, finalize evaluation results and write them to disk.
        Return the accuracy and the total number of test cases.
        """
        output_file_name = f"{VERSION_PREFIX}_{test_category}_score.json"
        output_file_dir = (
            score_dir / get_directory_structure_by_category(test_category)
        )
        write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    def _evaluate_single_agentic_entry(
        self,
        index,
        model_result_list,
        possible_answer_item,
        prompt_entry,
        test_category,
    ):
        """Helper method to process a single agentic entry."""
        # Remove the function doc from the score file for better readability
        if "function" in prompt_entry:
            del prompt_entry["function"]

        # Agentic test is a single-turn multi-step test, so the model result should be a list of one element
        if type(model_result_list) != list or len(model_result_list) != 1:
            return {
                "id": index,
                "test_category": test_category,
                "valid": False,
                "error": {
                    "error_message": [
                        "Error during inference phase. Model did not output a list of model responses."
                    ],
                    "error_type": "agentic:inference_error",
                },
                "prompt": prompt_entry,
                "model_result": model_result_list,
                "possible_answer": possible_answer_item,
            }

        model_answer = self._extract_answer(model_result_list, test_entry=prompt_entry)

        # Check if the model output contains the expected answer
        accuracy_checker_result = agentic_checker(
            model_answer,
            possible_answer_item,
        )

        if not accuracy_checker_result["valid"]:
            return {
                "id": index,
                "test_category": test_category,
                "valid": accuracy_checker_result.pop("valid"),
                "error": accuracy_checker_result,
                "prompt": prompt_entry["question"],
                "model_result_raw": model_result_list,
                "last_non_fc_message": last_unsuccessful_decoding_message,
                "possible_answer": possible_answer_item,
            }

        return {"valid": True}

    def _subset_entries_by_model_ids(
        self,
        model_result_entries: list[dict],
        prompt_entries: list[dict],
        ground_truth_entries: list[dict] = None,  # Irrelevance entries don't have ground truth
    ):
        """
        Filter the prompt and ground truth entries so that its order/length matches the IDs present in `model_result`. When `allow_missing` is False, all IDs must be present; otherwise, any missing IDs are silently ignored.
        """
        if not model_result_entries:
            return [], []

        if not self.allow_missing and (len(model_result_entries) != len(prompt_entries)):
            raise ValueError(
                f"Length of model result ({len(model_result_entries)}) does not match length of test entries ({len(prompt_entries)}). If you intended to run only on a subset (eg. entries present in the model result), please pass the `--partial-eval` flag."
            )

        all_present_ids = {entry["id"]: entry for entry in model_result_entries}

        # Align prompt and ground-truth using the *index* of the prompt entry. Some
        # ground-truth items use a different ID format, but the order between the
        # prompt list and the ground-truth list is guaranteed to be identical. We
        # therefore keep the element at index *i* in both lists whenever the
        # prompt entry at that index has an ID present in the model results.
        filtered_prompt_entries: list[dict] = []
        filtered_ground_truth_entries: list[dict] = []
        for idx, prompt_entry in enumerate(prompt_entries):
            if prompt_entry["id"] in all_present_ids:
                filtered_prompt_entries.append(prompt_entry)
                # ground_truth_entries and prompt_entries are aligned by index.
                if ground_truth_entries is not None:
                    filtered_ground_truth_entries.append(ground_truth_entries[idx])

        return filtered_prompt_entries, filtered_ground_truth_entries

    def _evaluate_single_relevance_entry(
        self,
        index,
        model_result_item,
        prompt_entry,
        test_category,
    ):
        """Helper method to process a single relevance/irrelevance entry."""
        contain_func_call = False
        decoded_result = None
        decode_error = None

        raise NotImplementedError("Relevance Not implemented yet.")

        try:
            decoded_result = self._decode_ast(
                model_result_item, language=ReturnFormat.PYTHON, has_tool_call_tag=False
            )
            # Decode successfully, which means the model output is in valid function call format
            contain_func_call = True
            if is_empty_output(decoded_result):
                # Empty output is not considered as a valid function call
                contain_func_call = False
        except Exception as e:
            # Decode failed, which means the model output is not in valid function call format
            contain_func_call = False
            decode_error = str(e)

        # irrelevance test means no function call outputted
        if "irrelevance" in test_category:
            success = not contain_func_call
        else:
            success = contain_func_call

        if not success:
            temp = {
                "id": index,
                "test_category": test_category,
                "valid": success,
                "prompt": prompt_entry,
                "model_result": model_result_item,
                "decoded_result": decoded_result,
            }
            if "irrelevance" in test_category:
                temp["error"] = ["Valid syntax. Successfully decode AST when it should not."]
                temp["error_type"] = "irrelevance_error:decoder_success"
            else:
                temp["error"] = [
                    f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"
                ]
                temp["error_type"] = "relevance_error:decoder_failed"
            return temp

        return {"valid": True}

    def _evaluate_single_multi_turn_entry(
        self,
        test_entry_id,
        model_result_list,
        ground_truth_list,
        prompt_entry,
        test_category,
    ):
        raise NotImplementedError("Multi Turn Not implemented yet.")

        """Helper method to process a single multi-turn entry."""
        # Remove the function doc from the score file for better readability
        if "function" in prompt_entry:
            del prompt_entry["function"]

        if type(model_result_list) != list:
            return {
                "id": test_entry_id,
                "test_category": test_category,
                "valid": False,
                "error": {
                    "error_message": [
                        "Error during inference phase. Model did not output a list of model responses."
                    ],
                    "error_type": "multi_turn:inference_error",
                },
                "prompt": prompt_entry,
                "model_result": model_result_list,
                "possible_answer": ground_truth_list,
            }

        # Check if force-terminated during inference phase.
        # This happens when the model has retried too many times and still haven't figured out the answer.
        # When force-terminated, no further evaluation is needed. This whole entry will be failed.
        if len(model_result_list) != len(ground_truth_list):
            return {
                "id": test_entry_id,
                "test_category": test_category,
                "valid": False,
                "error": {
                    "error_message": [
                        f"Model was force-terminated during inference phase. The length of the model result turns ({len(model_result_list)}) does not match the length of the ground truth turns ({len(ground_truth_list)})."
                    ],
                    "error_type": "multi_turn:force_terminated",
                },
                "prompt": prompt_entry,
                "model_result": model_result_list,
                "possible_answer": ground_truth_list,
            }

        # decode_execute returns a list of strings
        multi_turn_model_result_list_decoded: list[list[list[str]]] = []
        # Try decoding the model results into executable function calls
        for single_turn_model_result_list in model_result_list:
            single_turn_model_result_list_decoded = []
            for model_result_item in single_turn_model_result_list:
                # model_result_item is per step
                try:
                    decoded_result: list[str] = self._decode_execute(
                        model_result_item, has_tool_call_tag=False
                    )
                    if is_empty_execute_response(decoded_result):
                        # Empty output is not considered as a valid function call
                        continue
                    single_turn_model_result_list_decoded.append(decoded_result)
                except Exception as e:
                    # Ignore any failed decoding and continue to the next message
                    # We only care about the decoded function call, not the error message or if the model is chatting
                    continue
            multi_turn_model_result_list_decoded.append(single_turn_model_result_list_decoded)

        # Check if the model output the correct function calls
        accuracy_checker_result = multi_turn_checker(
            multi_turn_model_result_list_decoded,
            ground_truth_list,
            prompt_entry,
            test_category,
        )

        if not accuracy_checker_result["valid"]:
            return {
                "id": test_entry_id,
                "test_category": test_category,
                "valid": accuracy_checker_result.pop("valid"),
                "error": accuracy_checker_result,
                "prompt": prompt_entry,
                "model_result_raw": model_result_list,
                "model_result_decoded": multi_turn_model_result_list_decoded,
                "possible_answer": ground_truth_list,
            }

        return {"valid": True}

    def _evaluate_single_ast_entry(
        self,
        index,
        model_result_item,
        possible_answer_item,
        prompt_entry,
        test_category,
        language: Language,
        return_format: ReturnFormat,
        has_tool_call_tag=False,
    ):
        raise NotImplementedError("AST Not implemented yet.")
        """Helper method to process a single AST entry."""
        prompt_function = prompt_entry["function"]

        model_result_item_raw = model_result_item
        try:
            model_result_item = self._decode_ast(
                model_result_item, return_format, has_tool_call_tag
            )
        except Exception as e:
            return {
                "id": index,
                "test_category": test_category,
                "valid": False,
                "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                "error_type": "ast_decoder:decoder_failed",
                "prompt": prompt_entry,
                "model_result_raw": model_result_item_raw,
                "possible_answer": possible_answer_item,
            }

        decoder_output_valid = is_function_calling_format_output(model_result_item)
        if not decoder_output_valid:
            return {
                "id": index,
                "test_category": test_category,
                "valid": False,
                "error": [
                    "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                ],
                "error_type": "ast_decoder:decoder_wrong_output_format",
                "prompt": prompt_entry,
                "model_result_raw": str(model_result_item_raw),
                "model_result_decoded": str(model_result_item),
                "possible_answer": possible_answer_item,
            }

        checker_result = ast_checker(
            prompt_function,
            model_result_item,
            possible_answer_item,
            language,
            # format sensitivity has parallel, multiple cases which is encoded in index
            test_category if test_category != 'format_sensitivity' else index.split(':')[-1],
        )

        if not checker_result["valid"]:
            return {
                "id": index,
                "test_category": test_category,
                "valid": checker_result["valid"],
                "error": checker_result["error"],
                "error_type": checker_result["error_type"],
                "prompt": prompt_entry,
                "model_result_raw": model_result_item_raw,
                "model_result_decoded": model_result_item,
                "possible_answer": possible_answer_item,
            }
        return {"valid": True}

    def _extract_answer(self, model_result_list: list[str], test_entry):
        # TODO: 
        for model_result in model_result_list:
            if 'Final Answer:' in model_result:
                model_answer = model_result.split(
                    'Final Answer:'
                )[-1].strip()

            try:
                model_answer = eval(model_answer)
            except:
                pass

            if isinstance(model_answer, dict):
                if 'answer' in model_answer:
                    model_answer = model_answer['answer']
            return model_answer

        return model_result

    def _decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result, has_tool_call_tag)


    def _decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)
