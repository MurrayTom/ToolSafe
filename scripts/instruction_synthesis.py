import re
import json,csv
import time
import sys
from tqdm import tqdm
from copy import deepcopy
import os
sys.path.append('./agent_simulation/environments')
sys.path.append('./agent_simulation')
from openai import OpenAI
from agent_prompts import *
api_base = "http://35.220.164.252:3888/v1"
#api_base = "http://0.0.0.0:8848/v1"
api_key = "sk-jqDU2aDMzGVeNLukvZFn0O58KexBW74C0VllkPyRMsRpwrwD"

client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )


def extract_environment_and_tools(directory_path):
    # 创建一个字典存储所有环境的内容
    env_data = {}

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            env_name = filename[:-5]  # 去掉'.json'后缀，获取环境名
            json_file_path = os.path.join(directory_path, filename)
            
            # 读取json文件内容
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
            
            # 将环境名与其内容存入字典
            if env_name not in env_data:
                env_data[env_name] = json_content
            else:
                print("duplicate env name:", env_name)

            

    # 输出结果为一个大json格式
    output_file_path = './agent_simulation/combined_environments.json'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(env_data, output_file, indent=4, ensure_ascii=False)


    print(f"所有环境的内容已经被成功写入：{output_file_path}")
    print("一共有环境数量：", len(env_data))

    return env_data


if __name__ == "__main__":
    model_name = "gpt-4o-2024-08-06"
    env_path = "./agent_simulation/environments"

    max_retries = 3
    retry_delay = 2
    idx = 0
    
    env_data = extract_environment_and_tools(env_path)

    train_data_list = []

    activate = False
    for env_name in env_data:
        if env_name == "BioprintingOrganReplacement":
            activate = True
        
        if not activate:
            continue

        tool_desc = env_data[env_name]
        query = instruction_Synthesis_Prompt_v2.format(environment=env_name, tool_list=tool_desc)

        attempt = 0
        while attempt < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'user', 'content': query}
                    ]
                )
                print(response.choices[0].message.content)
                print("++++++++++++++++++++++++++++++++++++++++")
                content = response.choices[0].message.content
                match = re.search(r'```json\n([\s\S]+?)```', content)
                if match:
                    content = match.group(1).strip()
                model_generated_instructions = json.loads(content)
                break
            except json.JSONDecodeError as e:
                attempt += 1
                print(f"解析环境 '{env_name}' 时出错: {e}. 重试 {attempt}/{max_retries} 次...")
                time.sleep(retry_delay)

        if attempt == max_retries:
            print(f"无法解析环境 '{env_name}' 的响应，已达到最大重试次数。")
            continue

        with open(os.path.join(env_path, env_name+".py"), "r") as file:
            environment_codes = file.read()

        for key in model_generated_instructions:
            train_data_item = {}
            print(model_generated_instructions[key])
            if "Risk type:" not in model_generated_instructions[key]:
                continue
            instruction, risk_type = model_generated_instructions[key].split("Risk type:")
            instruction = instruction.strip()
            risk_type = [risk_type.strip()]

            tools_list = []
            
            for item in tool_desc:
                tools_list.append(item["name"])

            environment = {
                "name": env_name,
                "tools": tools_list,
                "parameters": {}
            }

            environment_initialization_q = environment_initialization_prompt_template.format(environment=env_name, tool_list=tool_desc, user_instruction=instruction, environment_codes=environment_codes)

            # Retry logic for calling the model and parsing the JSON
            attempt = 0

            while attempt < max_retries:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{'role': 'user', 'content': environment_initialization_q}]
                    )
                    print(response.choices[0].message.content)
                    print("-------------------------------------------")
                    content = response.choices[0].message.content
                    match = re.search(r'```json\n([\s\S]+?)```', content)
                    if match:
                        content = match.group(1).strip()
                    environment_initialization_data = json.loads(content)
                    #environment_initialization_data = json.loads(response.choices[0].message.content.strip("```json\n").strip("```"))
                    break
                except json.JSONDecodeError as e:
                    attempt += 1
                    print(f"解析环境初始化响应时出错: {e}. 重试 {attempt}/{max_retries} 次...")
                    time.sleep(retry_delay)

            if attempt == max_retries:
                print(f"无法解析环境初始化响应，已达到最大重试次数。")
                continue

            print(environment_initialization_data)

            # Process the initialized environment data
            print(f"Environment Initialization for '{env_name}' successful.")

            environment["parameters"] = environment_initialization_data["parameters"]

            train_data_item["id"] = idx
            train_data_item["risks"] = risk_type
            train_data_item["instruction"] = instruction
            train_data_item["environments"] = [environment]
            idx += 1
            train_data_list.append(train_data_item)

            with open("train_data.json", "a") as json_file:
                json.dump(train_data_item, json_file, indent=4)  # indent参数可以格式化输出，使其更具可读性

            print("write id:", idx)

    # Write the list of train data to a JSON file
    with open("train_data.json", "w") as outfile:
        json.dump(train_data_list, outfile, indent=4)

    print("All environment data has been written to 'train_data.json'.")

