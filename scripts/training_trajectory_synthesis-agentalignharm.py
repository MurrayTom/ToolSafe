import re
import json,csv
import pandas as pd
import yaml
import sys
from tqdm import tqdm
from copy import deepcopy
import os
import shutil
sys.path.append('./agent_simulation')
from openai import OpenAI
from agent_prompts import *
from synthetic_tools import AVAILABLE_TOOLS
api_base = "http://35.220.164.252:3888/v1"
#api_base = "http://0.0.0.0:8848/v1"
api_key = "sk-jqDU2aDMzGVeNLukvZFn0O58KexBW74C0VllkPyRMsRpwrwD"


client = OpenAI(
    api_key=api_key,
    base_url=api_base
)


# known_actions = {
#     "calculate": calculate,
#     "average_dog_weight": average_dog_weight
# }

# available_tool_descriptions = {
#     "calculate": "calculate the result of the expression",
#     "average_dog_weight": "get the average weight of a dog"
# }

# available_tool_params = {
#     "calculate": {
#         "what": "the expression to calculate"
#     },
#     "average_dog_weight": {
#         "name": "the breed of the dog"
#     }
# }

def parse_envs(envs_info):
    tool_list = []
    for tool in envs_info:
        if tool["type"] == "function":
            tool_list.append(tool["function"]["name"])
        else:
            print("singular tool:", tool)
    return tool_list

def format_tools_for_prompt(tool_descriptions):
    result = {}
    for tool in tool_descriptions:
        func = tool["function"]
        name = func["name"]
        desc = func["description"]
        params = func["parameters"]["properties"]
        converted_params = {}
        for k, v in params.items():
            param_dict = {"type": v.get("type", "string"), "description": v.get("description", "")}
            if "items" in v:
                param_dict["items"] = {"type": v["items"].get("type", "string")}
                if "description" in v["items"]:
                    param_dict["items"]["description"] = v["items"]["description"]
            converted_params[k] = param_dict
        result[name] = {"description": desc, "parameters": converted_params}
    
    result_yaml = yaml.dump(result, sort_keys=False, allow_unicode=True)
    
    return result_yaml

def extract_tool_params(text):
    """
    从模型输出中提取 tool_name 和 tool_params。
    支持以下两种格式：
        Action: average_dog_weight
        Action Input: breed="Bulldog"
    或：
        Action: average_dog_weight
        Action Input: {"breed": "Bulldog"}
    """
    pattern = r"Action:\s*(\w+).*?Action Input:\s*(.*)"
    match = re.search(pattern, text, re.S)
    if not match:
        return "", {}

    tool_name = match.group(1).strip()
    raw_params = match.group(2).strip()

    # 尝试 JSON 解析
    try:
        # 处理 JSON 坏格式，比如末尾多句号、反引号等
        raw_params_clean = raw_params.strip().rstrip('.').strip('`')
        params_dict = json.loads(raw_params_clean)
        if isinstance(params_dict, dict):
            return tool_name, params_dict
    except json.JSONDecodeError:
        pass  # 不是 JSON，就进入下一步解析

    # 尝试从形如 breed="Bulldog", age=2 的文本解析
    params_dict = {}
    pairs = re.findall(r'(\w+)\s*=\s*["\']?([^,"\']+)["\']?', raw_params)
    for k, v in pairs:
        params_dict[k] = v.strip()

    return tool_name, params_dict

class Agent:
    def __init__(self, system="", model_name="gpt-4o"):
        self.system = system
        self.messages = []
        self.model_name = model_name
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )
        return response.choices[0].message.content

def react_agent_invoke(model_name="gpt-4o-mini", query="", available_tool_descriptions=[], available_tool_list=[], max_turns=5):
    i = 0
    logs = ""
    available_tool_descriptions_str=format_tools_for_prompt(available_tool_descriptions)
    available_tool_list_str = str(available_tool_list)
    logs += "[System Prompt]:\n" + REACT_SYSTEM_PROMPT.format(available_tool_descriptions=available_tool_descriptions_str, available_tool_names=available_tool_list_str)
    logs += "\n-----------------------------------\n"
    bot = Agent(system=REACT_SYSTEM_PROMPT.format(available_tool_descriptions=available_tool_descriptions_str, available_tool_names=available_tool_list_str), model_name=model_name)
    next_prompt = query
    logs += query
    logs += "\n-----------------------------------\n"

    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        # print(result)
        # print("-----------------------------")
        logs += result
        logs += "\n-----------------------------------\n"

        tool_name, tool_params = extract_tool_params(result)

        if tool_name:
            #print(f" -- running {tool_name} {tool_params}")
            logs += f" -- running {tool_name} {tool_params}\n"
            if tool_name in available_tool_list:
                #observation = known_actions[tool_name].call_tool(tool_name, deepcopy(tool_params))
                try:
                    observation = AVAILABLE_TOOLS[tool_name](**tool_params)
                except Exception as e:
                    observation = f"calling {tool_name} an error occur: {str(e)}"
            else:
                observation = f"Unknown tool: {tool_name}"
            #print("Observation:", observation)
            logs += "Observation: "+ str(observation)
            logs += "\n-----------------------------------\n"

            next_prompt = f"Observation: {observation}"
        else:
            #print("Response:", result)
            break
    return logs, bot.messages

def filter_data(data_dir_path, log_csv_path):
    output_dir = data_dir_path + "_filter"
    os.makedirs(output_dir, exist_ok=True)

    matched_files = []

    for filename in os.listdir(data_dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if "Observation:" in content:
                matched_files.append(file_path)
                shutil.copy(file_path, os.path.join(output_dir, filename))
                print(f"✅ save: {filename}")

    print(f"filter completed, results in: {output_dir}")
    
    df = pd.read_csv(log_csv_path)

    if "log_path" not in df.columns:
        raise ValueError("CSV not found 'log_path' columns!")

    filtered_df = df[df["log_path"].isin(matched_files)]

    # 5. 保存到新的 CSV
    filtered_csv_path = os.path.splitext(log_csv_path)[0] + "_filtered.csv"
    filtered_df.to_csv(filtered_csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ filtered CSV: {filtered_csv_path}")



if __name__ == "__main__":
    model_name = "gpt-4o-2024-08-06"

    data_path = "./agent_simulation/data/agent_align_data_v3_harmful.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("the number of samples in seed dataset:", len(data))

    output_save_dir = "./agent_simulation/logs_agentalign_harm/{model_name}_{agent_type}".format(model_name=model_name, agent_type="react")
    os.makedirs(output_save_dir, exist_ok=True)

    output_csv_path = "./agent_simulation/logs_agentalign_harm/{model_name}_{agent_type}_.csv".format(model_name=model_name, agent_type="react")

    # # === 1. 初始化 CSV 文件（写表头）===
    csv_headers = ["id", "instruction", "category", "sub_category", "pattern",
                   "tools", "messages", "log_path"]
    
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()

    # === 2. 循环处理每个样本(进入agent) ===
    idx = 0
    for sample in data:
        idx += 1
        # if idx < 1910:
        #     continue
        output_save_path = output_save_dir + "/log_{id}.txt".format(id=sample["id"])
        user_query = sample["instruction"]

        tool_list = parse_envs(sample["tools"])
    
        logs, messages = react_agent_invoke(
            model_name=model_name, 
            query=user_query, 
            available_tool_descriptions=sample["tools"], 
            available_tool_list = tool_list
        )

        with open(output_save_path, 'a', encoding='utf-8') as f:
            f.write(logs)
            f.write("\n-----------------------------------\n")
        
        # === 3. 写入 CSV 一行 ===
        with open(output_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow({
                "id": sample.get("id", ""),
                "instruction": sample.get("instruction", ""),
                "category": sample.get("category", ""),
                "sub_category": sample.get("sub_category", ""),
                "pattern": sample.get("pattern", ""),
                "tools": json.dumps(sample.get("tools", ""), ensure_ascii=False),
                "messages": json.dumps(messages, ensure_ascii=False),
                "log_path": output_save_path
            })
        print(f"[{idx}/{len(data)}] ✅ Finished sample id={sample.get('id', '')}")

    # === 全部完成后打印一句 ===
    print(f"\n🎉 All {len(data)} samples processed successfully!")
    print(f"Results saved to: {output_csv_path}")

    ############# 处理为json文件 #######################





    ############# ------------- #######################

    filter_data(output_save_dir, output_csv_path)