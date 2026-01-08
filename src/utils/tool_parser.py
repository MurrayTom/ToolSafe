import re
import json

from utils.tool_parser import *
# from inspect_ai.log import transcript
# from inspect_ai.model import ChatMessage, ChatMessageUser, GenerateConfig, Model


# def extract_observation_react(text: str) -> str:
#     """
#     从字符串中提取 Observation: 后面的内容
#     """
#     match = re.search(r"Observation:\s*(.*)", text, re.DOTALL)
#     return match.group(1).strip() if match else ""


def extract_tool_params_react(text):
    """
    从模型输出中提取 tool_name 和 tool_params。
    支持以下两种格式：
        Action: average_dog_weight
        Action Input: breed="Bulldog"
    或：
        Action: average_dog_weight
        Action Input: {"breed": "Bulldog"}
    """
    text = str(text)
    text = re.sub(r'```json', '', text)
    pattern = r"Action:\s*(\w+)[\s\S]*?Action Input:\s*(\{.*\})"
    #pattern = r"Action:\s*(\w+).*?Action Input:\s*(.*)"
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

def extract_tool_params_planexecute(tool_call):
    # 1. 若为字符串，尝试解析为 JSON
    if isinstance(tool_call, str):
        try:
            tool_call = json.loads(tool_call)
        except json.JSONDecodeError:
            return "", {}

    # 2. 若不是 dict，直接返回空
    if not isinstance(tool_call, dict):
        return "", {}

    # 3. 正常字段抽取
    if "function_name" not in tool_call and "args" not in tool_call:
        return "", {}
    elif "function_name" not in tool_call:
        return "", tool_call.get("args", {})
    elif "args" not in tool_call:
        return tool_call.get("function_name", ""), {}
    else:
        return tool_call.get("function_name", ""), tool_call.get("args", {})


tool_extractor = {
    "react": extract_tool_params_react,
    "react_firewall": extract_tool_params_react,
    "plan_and_execute": extract_tool_params_planexecute,
    "sec_react": extract_tool_params_react
}

# observation_extractor = {
#     "react": extract_observation_react,
#     "sec-react": extract_observation_react
# }
